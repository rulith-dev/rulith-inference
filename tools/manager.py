"""Native Jan IPC helper. Reads one JSON request on stdin; emits one JSON result.

No listening socket, shell command input, or arbitrary executable selection.
Model weights are read only. Only the pinned strixllama runtime can be managed.
"""
import ctypes
from ctypes import wintypes
import datetime as dt
import errno
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'config' / 'jan'
# The one runtime: the pinned pwilkin llama.cpp branch built against the TheRock ROCm SDK in
# toolchain/rocm-venv. bootstrap/bootstrap.py produces it. Its folder was bin/hip-rocm101 up to 0.2.4, named
# after the ROCm 10.1 it was first built with; a tree that has only that folder still runs from it.
RUNTIME = next((p for p in (ROOT / 'bin' / d / 'llama-server.exe' for d in ('hip', 'hip-rocm101')) if p.is_file()),
               ROOT / 'bin' / 'hip' / 'llama-server.exe')
ROCM_BIN = ROOT / 'toolchain' / 'rocm-venv' / 'Lib' / 'site-packages' / '_rocm_sdk_devel' / 'bin'
# The author's launcher gates. LLAMA_MMB_HC16 must stay 0: with it on, Windows/TheRock/Clang
# floods the output with "/" (bisected in pwilkin/llama.cpp#24, reproduced here 2026-09-13).
HIP_GATES = dict(
    LLAMA_MMB=1, LLAMA_MMB_MIN_T=512, LLAMA_MMB_BF16W=1, LLAMA_MMB_GLU=1, LLAMA_MMB_TALL=2,
    LLAMA_MMB_CACHE=4, LLAMA_MMB_F32SPLIT=2, LLAMA_MMB_HC16=0, LLAMA_MMB_SHADOW=2, LLAMA_MMB_DOWN16=1,
    LLAMA_HC_CN_SHAPE=1, LLAMA_HC_GATEMIX=1, LLAMA_HC_MIX_FUSE=1, LLAMA_HC_BLK16=1,
    LLAMA_HC_RES16=1, LLAMA_HC_PACK_DI=1,
    LLAMA_NORM_GATED=1, LLAMA_NORM_ROWS=1, LLAMA_IDX_RELU_SUM=1, LLAMA_PLE_CONV=1, LLAMA_GDN_CONV=1,
    # The MTP draft context copies the target's batch sizes, so its compute buffers grow with the
    # target ubatch: at ctx 262144 / ubatch 8192 it asks for 3488 MiB and the load dies with
    # "cudaMalloc failed: out of memory". Capping the draft alone keeps both (patches/apply_spec_draft_ubatch.py).
    STRIX_SPEC_DRAFT_UBATCH=2048,
    # IndexShare for the MTP draft (patches/apply_mtp_index_share_031.py): once the draft's view of a
    # conversation passes 32K cells, a draft step attends to the sparse selection a catch-up kept -
    # refreshed every 32 positions - plus the cells since, instead of reading the whole cache. Per
    # speculative pass: -3.6% at 86K, -6.5% at 212K, same acceptance; below 32K nothing changes.
    LLAMA_MTP_INDEX_SHARE=1,
    # The MTP draft head's indexer cache (the draft context becomes a QSA memory). A ubatch of the
    # draft without outputs - a prompt, the catch-up after a verify - stores K, V and the indexer keys
    # and attends to nothing (its attention output reaches no output row), so the draft's prefill is no
    # longer quadratic; IndexShare above reads the selections. It cannot change what gets drafted
    # without IndexShare: the K/V stored are projections of the layer input, not of the attention
    # output. Does nothing unless the profile has MTP on.
    LLAMA_MTP_QSA=1)
# The sparse-attention gates, driven by the per-model QSA switch. LLAMA_QSA_BLOCK_SELECTION and
# LLAMA_QSA_DIRECT_INDICES together admit the block-selection path; LLAMA_QSA_SPARSE decides whether
# attention is restricted to the selected blocks or stays dense. QUERY_STRIP is a size, not a flag:
# 0 means "no strip", so it is left in place when the rest are off.
# This branch has no context threshold of its own - it engages once n_kv exceeds indexer_top_k, so
# the Jan "QSA 启用门槛" setting does not apply to the HIP runtime.
HIP_QSA_GATES = dict(
    LLAMA_QSA_SPARSE=1, LLAMA_QSA_BLOCK_SELECTION=1, LLAMA_QSA_COMPACT_METADATA=1,
    LLAMA_QSA_DENSE_SHORTCUT=1, LLAMA_QSA_DIRECT_INDICES=1, LLAMA_QSA_FA_V3=1, LLAMA_QSA_FUSE_EXPAND=1,
    LLAMA_QSA_NO_DENSE_MASK=1, LLAMA_QSA_PACK_KEYS=1, LLAMA_QSA_PACK_VALUES=1,
    LLAMA_QSA_SCORE_BOUNDS=1, LLAMA_QSA_WHOLE_ATTN=1,
    # decode-sized batches gather the selected cells instead of reading the whole cache: at 97K context
    # 49.3 vs 59.0 ms/token, and the context slope drops from 0.188 to 0.067 ms per 1000 tokens
    LLAMA_QSA_DECODE_GATHER=1,
    # Block-key cache. Safe with image input since patches/apply_qsa_kb_image_guard.py: the memory
    # reports whether it holds image cells and the graph stops wiring the cache in for that
    # conversation only, so a text-only chat keeps the cache (worth ~10% of decode).
    LLAMA_QSA_BLOCK_KEY_CACHE=1)
# The MTP draft and the attention work are specific to this model family, so the defaults that
# depend on them key off the file's name rather than one machine's path to it.
MODEL_FAMILY = 'Qwen3.8-Flash-Next'
# Unsloth's shared-Q4_K_M MTP head plus its own IQ4_XS copy of the LM head (tools/make_draft_head.py).
# The shared-* files borrow the target's Q6_K output.weight, 521 MB streamed on every draft step; the
# IQ4_XS copy is 338 MB. Every drafted token is verified by the target, so a coarser draft can only
# cost acceptance, and neither change did: Chinese 61% acceptance on the Q8_0 and the Q4_K_M base
# alike, English within its run-to-run spread (68-78% across configurations on one prompt).
# Q4_K_M over Q8_0 is worth 0.3-0.9 ms of a ~86 ms pass, i.e. under 1%; it is here for the 880 MB.
DEFAULT_DRAFT = ROOT / 'models' / 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf'
# Vision projector shipped alongside the model (clip, projector_type qwen3vl, 904 MB F16). Without it
# llama-server has no multimodal capability at all and rejects any request carrying an image.
MMPROJ_NAME = 'mmproj-F16.gguf'
# Thinking depth -> the reasoning_effort this model's chat template accepts. 'off' is handled
# separately (enable_thinking=false). The template raises on anything outside low/medium/xhigh.
THINKING = {'off': None, 'low': 'low', 'medium': 'medium', 'high': 'xhigh'}
# Where the server listens (Configuration › Network, GitHub issue #6): settings.json's `network` - the port, whether
# other devices on the local network may use it, the API key it then asks for - over these defaults, and RULITH_PORT,
# RULITH_HOST and RULITH_API_KEY over both (network()). settings.json stays where it is when the app updates.
NETWORK_DEFAULTS = dict(port=8080, lan=False, api_key='')
PORT_RANGE = (1024, 65535)
LOOPBACK, ANY_ADDRESS = '127.0.0.1', '0.0.0.0'
# the disk tier of the server's prompt cache: the default ceiling for config/jan/prompt-cache, and
# what a profile that predates the setting gets. A long conversation of this model is several GiB
# (79K tokens = 5.6 GiB, most of it context checkpoints), so this holds about three of them.
PROMPT_CACHE_DISK_MIB = 16384
# the RAM tier above it (--cache-ram). llama-server's default is 8192 MiB, which on this machine is a
# quarter of the system memory the GPU carve leaves - the KV cache itself lives in the carve, and this
# was the ~8 GB that came back when the model was unloaded. 1 GiB keeps short conversations resident;
# anything larger goes to the disk tier alone (prompt_save falls back to it when the RAM tier declines)
PROMPT_CACHE_RAM_MIB = 1024
# for a model the disk tier keeps as a whole state (version 2), how far a conversation grows before it is written
# again; version 3 - this model - writes rows in runs of 4096 positions as they are computed, and takes this only
# as the switch that lets idle slots hand over the runs they filled while busy
PROMPT_CACHE_BLOCK_TOKENS = 4096
# context checkpoints: the recurrent state at a point of a conversation, 0.11 GB each on this model, kept in system
# RAM while the conversation is resident. The ones used most sit at the last answer's edges (0.3.2), taken where the
# state already stands, without an extra pass: where it started - a regenerate is sampled again there from the logits
# kept with it, and the next turn of a client that drops the reasoning or re-renders a tool call goes back there - and
# where the answer before it ended, for an edit of the last message. Older ones only serve a deeper rewind (an edited
# earlier message, an agent trimming old tool output), which without one is processed again from further back.
# llama-server keeps up to 32 at least 8192 tokens apart, 3.5 GB a slot. This keeps 8, at most 0.9 GB a slot, chosen by
# the runtime: the last user message's start, the only place a prompt is cut for one (0.4.8; turn starts were cut
# from 0.2.6), and where batches end anyway, spaced by at least CHECKPOINT_MIN_STEP and a quarter of the distance to
# the end, dropped by what their loss would cost when the list is full (the last user message's never)
CTX_CHECKPOINTS = 8
CHECKPOINT_MIN_STEP = 4096
# the largest KV pool a profile may ask for (kv_pool): four full-length conversations. What actually fits is the
# GPU carve and the commit limit's business; this only stops a typo from asking for terabytes
KV_POOL_MAX = 1048576
# the largest KV pool the MTP draft keeps its full 2048-token ubatch for (see draft_ubatch)
DRAFT_UBATCH_FULL_POOL = 524288
# profile fields an earlier version had: a page or a saved profile that still sends one is not refused for it.
# shared_vram forced GGML_HIP_ENABLE_UNIFIED_MEMORY, which nothing reads (see runtime_environment).
RETIRED_FIELDS = ('shared_vram',)
# the K/V cache types the runtime's sparse attention takes (the profile's `kv`)
KV_TYPES = ('f16', 'q8_0')
# a loaded server is flagged (status()['commit_low']) when Windows has less commit than this left: four
# ~24K conversations took ~4 GB of it after the load, and two long ones keep up to ~3.5 GB of context
# checkpoints each in RAM
COMMIT_LOW_BYTES = 8 << 30
HIDDEN = 0x08000000 if os.name == 'nt' else 0
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ManagerError(ValueError):
    """An error the app can translate: `code` names it, `params` fill in the message. The text is
    English for the JSON callers; the pages render `errors.<code>` from their own locale."""
    def __init__(self, code, **params):
        self.code, self.params = code, params
        super().__init__(ERRORS[code].format(**params))


def fail(code, **params):
    raise ManagerError(code, **params)


ERRORS = {
    'not_gguf': 'Select a GGUF model file',
    'outside_roots': 'The model must be inside a registered model directory',
    'not_first_shard': 'Select the first shard of a split model',
    'gguf_truncated': 'The GGUF header is incomplete',
    'gguf_string': 'A GGUF string length is out of range',
    'gguf_nesting': 'GGUF arrays are nested too deep',
    'gguf_array': 'A GGUF array is too long',
    'gguf_type': 'Unknown GGUF field type',
    'gguf_version': 'Unsupported GGUF format',
    'gguf_count': 'The GGUF metadata count is out of range',
    'model_incomplete': 'The model is damaged or a shard is missing',
    'model_unlisted': 'The model is not in the list; rescan the model directories',
    'unknown_field': 'Unknown configuration field',
    'out_of_range': '{field} must be between {low} and {high}',
    'not_boolean': '{field} must be on or off',
    'thinking_level': 'Thinking depth must be one of {levels}',
    'draft_min': 'The MTP threshold must be between 0 and 1',
    'ubatch_gt_batch': 'ubatch cannot be larger than batch',
    'context_exceeds': 'The context is longer than the model declares',
    'kv_pool_below_context': 'The KV pool must hold at least one conversation of the full context ({context} tokens), or be 0',
    'kv_type': 'The KV cache is f16 or q8_0',
    'flash_attention_value': 'Flash Attention must be on or off',
    'draft_path': 'The draft model path is invalid',
    'mmproj_path': 'The vision projector path is invalid',
    'mmproj_missing': 'No vision projector ({name}) beside the model: add it, pick a file, or turn image input off',
    'qsa_architecture': 'Sparse attention (QSA) applies to Qwen3.8 Flash Next (qwen4exp) only',
    'qsa_needs_fa': 'Sparse attention (QSA) needs Flash Attention on',
    'mtp_architecture': 'MTP is enabled for qwen4exp models only',
    'draft_missing': "No MTP draft model found: put Unsloth's mtp-*.gguf beside the model, or turn MTP off",
    'head_mismatch': 'The draft head {head} does not belong to {base}',
    'identity_changed': 'The process identity has changed; refusing to unload it',
    'log_path': 'The log path is outside the project directory',
    'port_busy': 'Port {port} is in use by another service',
    'api_key_format': '{field} must be at most 256 printable ASCII characters, without spaces, commas or quotes',
    'host_address': 'RULITH_HOST must be an IPv4 address, such as 127.0.0.1 or 0.0.0.0',
    'host_unavailable': 'RULITH_HOST is {host}, which is not an address of this PC',
    'runtime_missing': 'The runtime is not there: {name}',
    'roots_count': 'Choose between 1 and 12 model directories',
    'root_missing': 'A model directory does not exist',
    'not_a_model': 'Drafts and vision projectors cannot be loaded as the chat model',
    'already_loaded': 'Unload the current model before loading another',
    'unknown_op': 'Unsupported operation',
    'request_too_large': 'The request is too large',
    'request_invalid': 'The request is malformed',
    'windows_only': 'This manager runs on Windows only',
    'stop_failed': 'The model process could not be stopped',
    'launch_failed': 'The model process did not start; see the log',
}
# why the last load ended, when its log said nothing more specific (status()['failure'])
FAILURES = {'oom': 'The GPU ran out of memory', 'error': 'The model process exited with an error'}
# The defaults are the measured configuration (docs/results.md), not a cautious one: context
# 262144, batch and ubatch 8192, flash attention on, and - per model, in profile() - sparse
# attention and MTP. On a carve this does not fit, the display driver places what is left in shared GPU
# memory by itself (measured at 64 GB: docs/results.md), slower but loaded.
DEFAULTS = dict(context=262144, gpu_layers=999, threads=16, batch=8192, ubatch=8192,
                # draft_max=3: swept again 2026-09-19 with the cheaper draft head, at 85K on real prose.
                # A fourth position costs 18.5 ms of an 86 ms pass (7.4 draft step + 11.2 target verify,
                # the latter being one more token's worth of expert bandwidth) and returned 0.56 tokens
                # per pass where it needed 0.64. 3 -> 27.9 ms/token, 4 -> 28.7, 5 -> 30.2.
                mtp=True, draft=str(DEFAULT_DRAFT), draft_max=3, draft_min=0.3,
                # ngram_spec off: it was measured free on prose and +27% on a coding turn, but those
                # runs had MTP off. Alongside the MTP draft it is neutral at best and costs real
                # throughput at depth - an 84K chat measured 25.50 tok/s with MTP alone against
                # 21.75 with both, the draft acceptance falling from 51% to 41% as the two drafters
                # compete for the same verification budget. On an 85K prose continuation it never
                # produced a draft at all (identical 183/229 counts with it on and off).
                # vision: load the multimodal projector, so requests may carry images. Costs 904 MB of
                # GPU memory and nothing else in this configuration - the two things llama-server turns
                # off when a projector is loaded, context shift and cache_reuse, are both already off
                # here (ctx_shift defaults false and this hybrid memory cannot shift anyway; we never
                # pass --cache-reuse). Ordinary prompt-prefix caching and MTP speculation are unaffected:
                # verified end to end, an image answered correctly with mtp on. Empty mmproj = auto.
                vision=True, mmproj='',
                # thinking: how hard the model reasons before answering, which this model's chat
                # template supports natively rather than us inventing it. 'off' sets
                # enable_thinking=false; the rest set reasoning_effort, which the template turns into
                # one injected system instruction. Its only legal values are low, medium and xhigh -
                # 'high' is an alias the template itself folds into xhigh, and 'medium' injects
                # nothing at all, i.e. the model's own default behaviour. Four levels is what this
                # model actually has; offering five would be two of them doing the same thing.
                # kv: the attention K/V cache type - f16, or q8_0 at half the memory (target K/V 6 -> 3.2 GiB at
                # 262144 tokens) and ~60% of the disk tier's bytes per token; the sparse-attention indexer keys and
                # the draft's cache stay f16 either way. The disk tier keeps only entries of the type in use.
                ngram_spec=False, kv='f16', flash_attention='on', thinking='off',
                qsa=False,
                # parallel: server slots, which are also the conversations that stay resident: talking to A, then B,
                # then A again finds A where it was, not processed again (with the disk tier off, every conversation
                # beyond the slots is). Each slot beyond the first costs ~0.43 GB of GPU memory (its recurrent state
                # rows; measured 75.8 / 76.2 / 77.1 / 78.7 GB at 1 / 2 / 4 / 8 slots, 2026-09-27) - the ~12 GB of
                # 0.1.x (a dense reserve for mixed-sequence batches) is long gone. They share one pool (-kvu), so one
                # conversation can still use the whole context while the others are idle, and a single conversation
                # decodes as it does on one slot. Several at once: 1/3/4 streams = 35.8/53.6/61.4 tok/s (0.2.4).
                # 8 since 0.2.7: agent tools run a session per sub-agent, and six agents on four slots pushed each
                # other out - every step processed again from the system prompt, 1.8x the tokens and the time to
                # first token 12 s instead of ~4 (tmp/ragged/agent_sim.py). A conversation in use also keeps up to
                # 8 snapshots of its recurrent state in RAM, ~0.9 GB (six agents: server working set 6.3 -> 9.8 GB).
                parallel=8,
                # kv_pool: the cells of the one KV pool that several slots share (-kvu), when it should hold more
                # than one conversation at full length: each conversation stays capped at `context`
                # (--kv-unified-per-slot) and the pool is one allocation of any size, rounded up to 256 cells.
                # 0 = the pool is the context. For this model every cell costs ~32.5 KiB of GPU memory -
                # 262144 cells: target K/V 6 GiB, indexer 0.75, block keys 0.75, the draft's 0.63 - and the
                # target's K/V is one allocation, which on this machine counts against Windows' commit limit.
                # Ignored with one slot, whose pool is its context.
                kv_pool=0,
                # trunk_decode_q6k: Q6_K in-memory copies of the Q8_0 trunk for decode-sized batches (HIP);
                # +2.9 GB VRAM, prefill untouched, decode -9%. Off by default so smaller carves still load.
                trunk_decode_q6k=False,
                # prompt_cache_disk: the server's prompt cache gets a disk tier (config/jan/prompt-cache,
                # PROMPT_CACHE_DISK_MIB). A conversation's attention rows - ~29 KB per token for this model,
                # draft included - are written once, 4096 positions at a time, as they are computed; its
                # recurrent state, the part that changes, only when it leaves memory: another conversation
                # needs its cells, or the server is stopped (stop has it write first). When the conversation
                # returns it is read back instead of processed - 33K tokens in 0.4 s against ~35 s of
                # prefill - and that survives restarts. Off by default since 0.1.13: nothing is written to the
                # SSD unless asked for (0.1.12's tier also moved whole states through RAM, 5 GB at 173K
                # tokens, which took this machine to its commit limit when two long conversations swapped:
                # GitHub issue #1).
                prompt_cache_disk=False,
                # prompt_cache_disk_mib: the ceiling for that directory. Oldest goes first once it is
                # reached, so the only cost of a larger number is disk; 200 GiB of a 1 TB drive keeps
                # every conversation this machine can hold.
                prompt_cache_disk_mib=PROMPT_CACHE_DISK_MIB)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


_LMSTUDIO = []


def lmstudio_models_dir():
    """LM Studio's download folder, if it is installed and the folder exists.

    This model is 93.7 GB and there is no reason to hold two copies of it. LM Studio lays models
    out as <root>/<publisher>/<repo>/<file>.gguf, which is exactly the shape catalog() walks, so
    adding its folder as a root makes everything already downloaded appear here — the same files,
    read-only, no import step and nothing copied.
    """
    if not _LMSTUDIO:
        found = None
        try:
            cfg = json.loads((Path.home() / '.lmstudio' / 'settings.json').read_text(encoding='utf-8'))
            folder = cfg.get('downloadsFolder')
            if folder and Path(folder).is_dir():
                found = str(Path(folder))
        except (OSError, ValueError):
            pass
        _LMSTUDIO.append(found)
    return _LMSTUDIO[0]


def settings():
    roots = [str(ROOT / 'models')]
    lms = lmstudio_models_dir()
    if lms:
        roots.append(lms)
    return read_json(DATA / 'settings.json', {'roots': roots, 'profiles': {}})


def api_key_valid(key):
    # llama-server reads --api-key as a comma-separated list (a comma would make two weaker keys) and parses it as
    # CSV, so no commas or quotes; and nothing a command line or an Authorization header would have to escape
    return len(key) <= 256 and all('!' <= c <= '~' and c not in ',"' for c in key)


def check_network(raw):
    """A network setting to save or use: port 1024-65535, lan a switch, api_key empty or one api_key_valid() takes.
    A field it does not know is refused, as validate_profile() refuses one."""
    if not isinstance(raw, dict) or set(raw) - set(NETWORK_DEFAULTS): fail('unknown_field')
    net = {**NETWORK_DEFAULTS, **raw}
    if type(net['port']) is not int or not PORT_RANGE[0] <= net['port'] <= PORT_RANGE[1]:
        fail('out_of_range', field='port', low=PORT_RANGE[0], high=PORT_RANGE[1])
    if type(net['lan']) is not bool: fail('not_boolean', field='lan')
    if not isinstance(net['api_key'], str) or not api_key_valid(net['api_key']): fail('api_key_format', field='api_key')
    return net


def saved_network():
    """settings.json's `network` over the defaults. A file from before the setting has none; a field a later version
    added is dropped rather than refused, as profile() drops one."""
    raw = settings().get('network')
    return check_network({k: v for k, v in raw.items() if k in NETWORK_DEFAULTS} if isinstance(raw, dict) else {})


def network():
    """The network setting in effect: saved_network(), with RULITH_PORT, RULITH_HOST (the address to bind, e.g.
    0.0.0.0) and RULITH_API_KEY over it. host is where the server binds - 0.0.0.0 with lan on, else 127.0.0.1 - and
    forced names the variable behind each field one sets, which the page then shows instead of offering it."""
    net, forced = saved_network(), {}
    port = os.environ.get('RULITH_PORT', '').strip()
    if port:
        if not re.fullmatch(r'\d{1,5}', port) or not PORT_RANGE[0] <= int(port) <= PORT_RANGE[1]:
            fail('out_of_range', field='RULITH_PORT', low=PORT_RANGE[0], high=PORT_RANGE[1])
        net['port'], forced['port'] = int(port), 'RULITH_PORT'
    host = ANY_ADDRESS if net['lan'] else LOOPBACK
    named = os.environ.get('RULITH_HOST', '').strip()
    if named:
        try: address = ipaddress.IPv4Address(named)
        except ValueError: fail('host_address')
        host, net['lan'], forced['lan'] = str(address), not address.is_loopback, 'RULITH_HOST'
    key = os.environ.get('RULITH_API_KEY', '').strip()
    if key:
        if not api_key_valid(key): fail('api_key_format', field='RULITH_API_KEY')
        net['api_key'], forced['api_key'] = key, 'RULITH_API_KEY'
    return {**net, 'host': host, 'forced': forced}


def identifier(path):
    return hashlib.sha256(str(Path(path).resolve()).casefold().encode()).hexdigest()[:20]


def checked_file(value):
    path = Path(value).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.gguf':
        fail('not_gguf')
    if not any(path.is_relative_to(Path(root).resolve()) for root in settings()['roots']):
        fail('outside_roots')
    if re.search(r'-(?!00001)\d{5}-of-\d{5}\.gguf$', path.name):
        fail('not_first_shard')
    return path


def mmproj_path(cfg, model):
    """The vision projector to load: the profile's own path, else the one beside THIS model.

    Beside the model actually being loaded, not beside a default one — a model kept in a second
    directory would otherwise silently load the first model's projector.
    """
    return Path(cfg['mmproj']) if cfg.get('mmproj') else Path(model['path']).parent / MMPROJ_NAME


def metadata(path):
    """Read GGUF metadata only, seeking past large tokenizer arrays."""
    with path.open('rb') as f:
        def exact(n):
            b = f.read(n)
            if len(b) != n:
                fail('gguf_truncated')
            return b
        def u32(): return struct.unpack('<I', exact(4))[0]
        def u64(): return struct.unpack('<Q', exact(8))[0]
        def string(keep=True):
            n = u64()
            if n > 16 * 1024 * 1024:
                fail('gguf_string')
            if keep:
                return exact(n).decode('utf-8', 'replace')
            f.seek(n, 1)
        fmt = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, keep=True, depth=0):
            if depth > 3: fail('gguf_nesting')
            if kind in fmt:
                code = '<' + fmt[kind]
                return struct.unpack(code, exact(struct.calcsize(code)))[0]
            if kind == 8: return string(keep)
            if kind == 9:
                item, n = u32(), u64()
                if n > 10_000_000: fail('gguf_array')
                if item in fmt:
                    f.seek(n * struct.calcsize('<' + fmt[item]), 1)
                else:
                    for _ in range(n): value(item, False, depth+1)
                return None
            fail('gguf_type')
        if exact(4) != b'GGUF' or u32() not in (2, 3):
            fail('gguf_version')
        tensors, count = u64(), u64()
        if count > 100000: fail('gguf_count')
        meta = {}
        for _ in range(count):
            key = string()
            keep = key.startswith(('general.', 'split.')) or key.endswith(('.context_length', '.block_count',
                # what gpu_need_bytes sizes the KV pool from
                '.attention.head_count_kv', '.attention.key_length', '.attention.value_length', '.full_attention_interval',
                '.attention.indexer.key_length'))
            val = value(u32(), keep)
            if keep and val is not None: meta[key] = val
        return meta


def gguf_layout(path):
    """The byte layout of a GGUF v3 file: the key-value + tensor-info span, the alignment, the
    tensors as (name, dims, type, offset) and where the data section starts. Enough to splice two
    files together without decoding a tensor, which is all merge_draft_head needs."""
    path = Path(path)
    with path.open('rb') as f:
        def exact(n):
            b = f.read(n)
            if len(b) != n: fail('gguf_truncated')
            return b
        def u32(): return struct.unpack('<I', exact(4))[0]
        def u64(): return struct.unpack('<Q', exact(8))[0]
        fmt = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, depth=0):
            if depth > 3: fail('gguf_nesting')
            if kind in fmt:
                code = '<' + fmt[kind]
                return struct.unpack(code, exact(struct.calcsize(code)))[0]
            if kind == 8:
                n = u64()
                if n > 16 * 1024 * 1024: fail('gguf_string')
                return exact(n).decode('utf-8', 'replace')
            if kind == 9:
                item, n = u32(), u64()
                if n > 10_000_000: fail('gguf_array')
                if item in fmt: f.seek(n * struct.calcsize('<' + fmt[item]), 1)
                else:
                    for _ in range(n): value(item, depth + 1)
                return None
            fail('gguf_type')
        if exact(4) != b'GGUF' or u32() != 3: fail('gguf_version')
        n_tensors, n_kv = u64(), u64()
        if n_kv > 100000 or n_tensors > 100000: fail('gguf_count')
        kv_start, alignment = f.tell(), 32
        for _ in range(n_kv):
            key = value(8); val = value(u32())
            if key == 'general.alignment' and isinstance(val, int): alignment = val
        infos_start, tensors = f.tell(), []
        for _ in range(n_tensors):
            name = value(8); ndim = u32()
            if ndim > 8: fail('gguf_count')
            dims = [u64() for _ in range(ndim)]
            tensors.append((name, dims, u32(), u64()))
        infos_end = f.tell()
    return dict(n_kv=n_kv, span=(kv_start, infos_end), alignment=alignment, tensors=tensors,
                data_start=(infos_end + alignment - 1) // alignment * alignment, size=path.stat().st_size)


def merge_draft_head(base, head, out):
    """Write `out` = the draft `base` (Unsloth's mtp-*-shared-*.gguf) with the tensors of `head`
    appended - the file tools/make_draft_head.py produces, made here from a downloaded head instead of
    a 50 GB target shard and a toolchain. A head is either the draft's own quantised LM head
    (output.weight, *-head-iq4_xs) or, since 0.4.6, the low-rank pre-score that lets the draft use the
    target's head for a few candidates only (blk.N.nextn.lr_proj / lr_scores, *-head-lr512). The
    base's key-value and tensor-info bytes are copied verbatim, and so is the head's data section:
    offsets are relative to the data section, whose alignment is kept, so nothing is re-encoded."""
    base, head, out = Path(base), Path(head), Path(out)
    b, h = gguf_layout(base), gguf_layout(head)
    names = [t[0] for t in h['tensors']]
    if (not names or any(t[0] in names for t in b['tensors']) or h['alignment'] != b['alignment']
            or (names != ['output.weight'] and any(not n.startswith('blk.') or '.nextn.lr_' not in n for n in names))
            or metadata(head).get('general.architecture') != metadata(base).get('general.architecture')):
        fail('head_mismatch', head=head.name, base=base.name)
    # a file with no tensors ends before its (aligned) data section would start, hence the clamp
    align, base_bytes = b['alignment'], max(0, b['size'] - b['data_start'])
    offset = (base_bytes + align - 1) // align * align
    info = b''.join(struct.pack('<Q', len(name.encode())) + name.encode() + struct.pack('<I', len(dims))
                    + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', ttype, offset + off)
                    for name, dims, ttype, off in h['tensors'])
    part = out.with_suffix('.part')
    with base.open('rb') as src, head.open('rb') as hd, part.open('wb') as dst:
        dst.write(b'GGUF' + struct.pack('<IQQ', 3, len(b['tensors']) + len(names), b['n_kv']))
        src.seek(b['span'][0]); dst.write(src.read(b['span'][1] - b['span'][0]))
        dst.write(info); dst.write(b'\0' * (-dst.tell() % align))
        src.seek(b['data_start'])
        for chunk in iter(lambda: src.read(16 << 20), b''): dst.write(chunk)
        dst.write(b'\0' * (offset - base_bytes))
        hd.seek(h['data_start'])
        for chunk in iter(lambda: hd.read(16 << 20), b''): dst.write(chunk)
    part.replace(out)
    return out


def draft_head_name(name):
    """mtp-<family>-head-<type>.gguf: a downloadable head, not a draft the loader could run."""
    low = name.lower()
    return low.startswith('mtp-') and '-head-' in low and '-shared-' not in low


def merge_heads(found):
    """For every downloaded head in the catalog whose family has Unsloth's shared draft in the same
    directory, make the merged draft once (<shared>-head-<type>.gguf) if it is not there yet.
    Returns (paths written, {head path: error})."""
    merged, errors = [], {}
    for head in found:
        if head.get('role') != 'head' or head.get('error'): continue
        low = head['filename'].lower()
        family, kind = head['filename'][4:low.index('-head-')], head['filename'][low.index('-head-') + 6:-5]
        folder = Path(head['path']).parent
        bases = [m for m in found if m.get('role') == 'draft' and not m.get('error') and Path(m['path']).parent == folder
                 and m['filename'].lower().startswith(f'mtp-{family}-shared-'.lower()) and '-head-' not in m['filename'].lower()]
        bases.sort(key=lambda m: ('Q4_K_M' not in m['filename'], m['filename']))
        if not bases: continue
        out = Path(bases[0]['path']).with_name(Path(bases[0]['path']).stem + f'-head-{kind}.gguf')
        if out.exists(): continue
        try: merged.append(str(merge_draft_head(bases[0]['path'], head['path'], out)))
        except (OSError, ValueError, struct.error) as exc: errors[head['path']] = str(exc)
    return merged, errors


def catalog(refresh=False, _after_merge=False):
    cached = read_json(DATA / 'catalog.json', None)
    if cached is not None and not refresh: return cached
    found = []
    seen = set()
    for root in settings()['roots']:
        for parent, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith('.') and not Path(parent, d).is_symlink()]
            for name in sorted(files):
                if not name.lower().endswith('.gguf') or re.search(r'-(?!00001)\d{5}-of-\d{5}\.gguf$', name): continue
                path = Path(parent, name).resolve()
                key = identifier(path)
                if key in seen: continue
                seen.add(key)
                try:
                    path = checked_file(path)
                    meta = metadata(path)
                    split = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', name)
                    shards = [path]
                    if split:
                        shards = [path.with_name(name[:split.start()] + f'-{i:05}-of-{int(split[2]):05}.gguf') for i in range(1, int(split[2])+1)]
                    missing = [p.name for p in shards if not p.exists()]
                    role = ('projection' if name.lower().startswith('mmproj') or meta.get('general.architecture') == 'clip'
                            else 'head' if draft_head_name(name) else 'draft' if name.lower().startswith('mtp-') else 'model')
                    found.append(dict(id=key, path=str(path), name=meta.get('general.name', path.stem),
                                      filename=name, architecture=meta.get('general.architecture', 'unknown'),
                                      size=sum(p.stat().st_size for p in shards if p.exists()), shards=len(shards),
                                      missing=missing, role=role, context=next((v for k,v in meta.items() if k.endswith('.context_length')), None),
                                      quant=re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I)[1]
                                      if re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I) else str(meta.get('general.file_type', ''))))
                except (OSError, ValueError, struct.error) as exc:
                    found.append(dict(id=key, path=str(path), filename=name, name=path.stem, error=str(exc), role='invalid', size=0))
    # a rescan is also when a downloaded draft head gets merged with its shared draft; the merged
    # file is then scanned like any other, once
    merged, errors = merge_heads(found) if refresh and not _after_merge else ([], {})
    if merged:
        result = catalog(True, _after_merge=True)
    else:
        result = {'models': sorted(found, key=lambda x: (x['role'], x['filename'])), 'roots': settings()['roots'], 'scanned_at': dt.datetime.now().astimezone().isoformat()}
    if refresh and not _after_merge:
        result.update(merged=merged, merge_errors=errors)
    atomic_json(DATA / 'catalog.json', result)
    return result


def model_by_id(model_id):
    for m in catalog()['models']:
        if m['id'] == model_id:
            if m.get('error') or m.get('missing'): fail('model_incomplete')
            checked_file(m['path'])
            return m
    fail('model_unlisted')


def family_draft():
    """The MTP draft for this family, if one is on disk: the repository's own first, then the best
    file under the model roots - a merged *-head-* draft (merge_draft_head) over Unsloth's
    shared-Q4_K_M over shared-Q8_0, all of which draft the same tokens. None when there is none, and
    then profile() leaves MTP off instead of pointing at a file that is not there."""
    if DEFAULT_DRAFT.is_file():
        return str(DEFAULT_DRAFT)
    drafts = [m['path'] for m in catalog()['models'] if m.get('role') == 'draft' and MODEL_FAMILY in m['filename'] and not m.get('error')]
    # the low-rank head (0.4.6) drafts the same tokens as the others, faster
    drafts.sort(key=lambda p: ('-head-' not in p.lower(), '-head-lr' not in p.lower(), 'Q4_K_M' not in p, p))
    return drafts[0] if drafts else None


def profile(model):
    default = dict(DEFAULTS)
    # MTP and image input default to on, but only when their file is actually there. A load that
    # refuses to start because a companion file is missing is the wrong first experience; the
    # switch turns itself on once the file appears and the directories are rescanned.
    default['draft'] = family_draft() or ''
    default['mtp'] = MODEL_FAMILY in Path(model['path']).name and bool(default['draft'])
    default['vision'] = mmproj_path({}, model).is_file()
    # sparse attention is this architecture's, and above 64K context it is what makes the load fit
    default['qsa'] = model.get('architecture') == 'qwen4exp'
    if model.get('context'): default['context'] = min(default['context'], int(model['context']))
    saved = settings()['profiles'].get(model['id'], {})
    # A profile written by an earlier version can carry fields this one no longer has. They are
    # dropped here rather than echoed to the page, which would send them straight back and have
    # validate_profile() refuse the whole profile as unknown.
    cfg = {**default, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    # a profile saved with an older merged head moves to the low-rank one once that is on disk (0.4.6): it
    # drafts the same tokens, faster, and nobody chose the older head over it
    if ('-head-lr' in Path(default['draft']).name.lower() and '-head-' in Path(cfg['draft'] or '').name.lower()
            and '-head-lr' not in Path(cfg['draft']).name.lower()):
        cfg['draft'] = default['draft']
    # thinking was a switch before it was a level. Normalise here and not only in
    # validate_profile(): this is what the configuration page displays, and a stored `true`
    # reached it as a level called "true" whose help text does not exist.
    if type(cfg['thinking']) is bool: cfg['thinking'] = 'high' if cfg['thinking'] else 'off'
    return cfg


def validate_profile(raw, model):
    if isinstance(raw, dict): raw = {k: v for k, v in raw.items() if k not in RETIRED_FIELDS}
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS): fail('unknown_field')
    cfg = {**profile(model), **raw}
    bounds = dict(context=(512,262144), gpu_layers=(0,999), threads=(1,32), batch=(32,32768), ubatch=(32,32768), draft_max=(1,8), parallel=(1,16),
                  prompt_cache_disk_mib=(1024,262144), kv_pool=(0,KV_POOL_MAX))
    for field, (low, high) in bounds.items():
        if type(cfg[field]) is not int or not low <= cfg[field] <= high: fail('out_of_range', field=field, low=low, high=high)
    for field in ('mtp', 'ngram_spec', 'qsa', 'trunk_decode_q6k', 'vision', 'prompt_cache_disk'):
        if type(cfg[field]) is not bool: fail('not_boolean', field=field)
    # thinking was a switch before it was a level; a profile saved back then still loads
    if type(cfg['thinking']) is bool: cfg['thinking'] = 'high' if cfg['thinking'] else 'off'
    if cfg['thinking'] not in THINKING: fail('thinking_level', levels=', '.join(THINKING))
    if type(cfg['draft_min']) not in (int,float) or not 0 <= cfg['draft_min'] <= 1: fail('draft_min')
    if cfg['ubatch'] > cfg['batch']: fail('ubatch_gt_batch')
    if model.get('context') and cfg['context'] > model['context']: fail('context_exceeds')
    if cfg['kv_pool'] and cfg['kv_pool'] < cfg['context']: fail('kv_pool_below_context', context=cfg['context'])
    if cfg['kv'] not in KV_TYPES: fail('kv_type')
    if cfg['flash_attention'] not in ('on', 'off'): fail('flash_attention_value')
    if not isinstance(cfg['draft'], str): fail('draft_path')
    if not isinstance(cfg['mmproj'], str): fail('mmproj_path')
    if cfg['vision']:
        # a path the user picked goes through the full check (it must sit in a registered model root);
        # the automatic one ships beside the model, so only its existence matters
        if cfg['mmproj']: checked_file(cfg['mmproj'])
        elif not mmproj_path(cfg, model).is_file(): fail('mmproj_missing', name=MMPROJ_NAME)
    if cfg['qsa']:
        if model.get('architecture') != 'qwen4exp':
            fail('qsa_architecture')
        if cfg['flash_attention'] != 'on': fail('qsa_needs_fa')
    if cfg['mtp']:
        if model['architecture'] != 'qwen4exp': fail('mtp_architecture')
        if not cfg['draft'] or not Path(cfg['draft']).is_file(): fail('draft_missing')
        checked_file(cfg['draft'])
    return cfg


def bundled_rocm():
    """The installed layout: the ROCm DLLs sit beside llama-server (tools/make_runtime_bundle.py),
    so Windows finds them without a PATH entry and there is no SDK directory at all. A build tree
    carries only the HIP runtime DLLs that System32 would otherwise shadow (bootstrap.py --build) and
    takes the rest of the SDK through PATH, so the test is a library only the bundle carries."""
    return (RUNTIME.parent / 'hipblas.dll').is_file()


def runtime_available():
    return (RUNTIME.is_file() and (RUNTIME.parent/'ggml-hip.dll').is_file()
            and (bundled_rocm() or (ROCM_BIN.is_dir() and (ROCM_BIN/'amdhip64_7.dll').is_file())))


def selected_runtime(cfg):
    return RUNTIME


def managed_runtime(path):
    # a server an install before 0.2.5 started from bin/hip-rocm101 is still this runtime's to adopt and stop
    legacy = RUNTIME.parent.parent / 'hip-rocm101' / RUNTIME.name
    return bool(path) and Path(path).resolve() in (RUNTIME.resolve(), legacy.resolve())


def runtime_info():
    """The ROCm release the runtime was built and bundled with, and the GPU target it carries kernels for:
    from BUNDLE.json in an install (tools/make_runtime_bundle.py), else from the SDK's own package names."""
    try:
        bundle = RUNTIME.parents[2] / 'BUNDLE.json'
        if bundle.is_file():
            b = json.loads(bundle.read_text(encoding='utf-8'))
            m = re.search(r'-(\d[\w.]*)\.dist-info$', b.get('rocm') or '')
            return {'rocm': m.group(1) if m else None, 'gfx': b.get('gfx')}
        site = ROCM_BIN.parents[1]
        core = sorted(site.glob('rocm_sdk_core-*.dist-info'))
        device = sorted(site.glob('rocm_sdk_device_gfx*.dist-info'))
        gfx = re.match(r'rocm_sdk_device_(gfx\w+?)-', device[-1].name) if device else None
        return {'rocm': core[-1].name[len('rocm_sdk_core-'):-len('.dist-info')] if core else None,
                'gfx': gfx.group(1) if gfx else None}
    except (OSError, ValueError, IndexError):
        return {}


def dedicated_vram_bytes():
    """The GPU's dedicated memory as the display driver registered it - on this machine, the BIOS
    carve. Read from the registry rather than asked of HIP, so it costs nothing and needs no GPU
    context. None when it cannot be read (not Windows, no adapter entry)."""
    try:
        import winreg
    except ImportError:
        return None
    best = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}') as adapters:
            for i in range(64):
                try: name = winreg.EnumKey(adapters, i)
                except OSError: break
                try:
                    with winreg.OpenKey(adapters, name) as adapter:
                        size = winreg.QueryValueEx(adapter, 'HardwareInformation.qwMemorySize')[0]
                except OSError: continue
                if isinstance(size, int) and size > (best or 0): best = size
    except OSError:
        return None
    return best


def draft_ubatch(pool):
    """The MTP draft's ubatch for a KV pool of `pool` cells: HIP_GATES' 2048 up to 512K cells, then smaller in
    proportion, in steps of 256, never under 512."""
    base = int(HIP_GATES['STRIX_SPEC_DRAFT_UBATCH'])
    if pool <= DRAFT_UBATCH_FULL_POOL:
        return base
    return max(512, base * DRAFT_UBATCH_FULL_POOL // pool // 256 * 256)


def kv_pool_cells(cfg):
    """The cells of the KV pool a load allocates: the context, or with several slots the kv_pool setting when it
    is larger, rounded up to the 256 cells llama.cpp pads the pool to."""
    pool = cfg['context']
    if cfg.get('parallel', 1) > 1 and cfg.get('kv_pool', 0) > pool:
        pool = (cfg['kv_pool'] + 255) // 256 * 256
    return pool


# Experts that do not fit the carve live in pinned system memory, read by the GPU in place through ROCm_Host (0.4.7).
# Left to the display driver, the overflow lands on whatever is allocated last - the KV cache and the compute buffers,
# which every token reads in small pieces: at the 64 GB carve decode ran 28% slower (09-18), at the 96 GB carve a KV
# pool past the carve cost 10-12% at long context. Experts are read whole, and from system memory as fast as from the
# carve (tmp/vram64: ~230 GB/s either way; at the 64 GB carve with 12-16 layers' experts there, prefill and decode
# matched the 96 GB carve). The estimate below sums what a load puts on the GPU, from the GGUF and the profile, with
# the per-slot and per-ubatch terms measured from the runtime's own buffer report (tmp/vram64, v64_cal); the experts
# of the last layers go to system memory until the rest fits, leaving HOST_EXPERT_MARGIN of the carve.
MIB = 1 << 20
CPU_TENSORS = ('token_embd.weight', 'per_layer_token_embd.weight')   # llama.cpp keeps these in system memory
HOST_EXPERT_MARGIN = 2560 * MIB          # context overhead outside the buffer report (~1.5 GiB) plus headroom
CARVE_USABLE = 0.96                      # the driver keeps the rest: at the 64 GB carve, dedicated tops out at 62.1 GiB
KV_ELEM_BYTES = {'f16': 2.0, 'bf16': 2.0, 'q8_0': 34 / 32, 'q5_1': 24 / 32, 'q5_0': 22 / 32, 'q4_1': 20 / 32, 'q4_0': 18 / 32,
                 'iq4_nl': 18 / 32}
RS_BYTES_PER_SLOT = 469 * MIB            # qwen4exp's delta-net and conv states, one conversation (3747 MiB for 8)
# the target's compute buffer grows with the ubatch and with the pool: 3522 MiB at 262144 cells, 5206 MiB at 512000
# (both ub 8192), i.e. ~1755 MiB for the ubatch and ~6.74 KiB a cell
COMPUTE_BYTES_PER_UBATCH_TOKEN = 1755 * MIB / 8192
COMPUTE_BYTES_PER_CELL = 6.74 * 1024
DRAFT_MASK_BYTES_PER_CELL = 4.8 * 1024   # at the draft's 2048-token ubatch (kv-pool-draft-mask)


def model_shards(model):
    path = Path(model['path'])
    split = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path.name)
    if not split: return [path]
    return [path.with_name(path.name[:split.start()] + f'-{i:05}-of-{int(split[2]):05}.gguf') for i in range(1, int(split[2]) + 1)]


def tensor_bytes(paths):
    """{tensor name: bytes in the file} over a model's shards, from consecutive offsets - no type table needed."""
    out = {}
    for path in paths:
        lay = gguf_layout(path)
        ts = sorted(lay['tensors'], key=lambda t: t[3])
        end = lay['size'] - lay['data_start']
        for i, t in enumerate(ts):
            out[t[0]] = (ts[i + 1][3] if i + 1 < len(ts) else end) - t[3]
    return out


def gpu_need_bytes(model, cfg, tb):
    """What a load of `model` with profile `cfg` puts on the GPU: weights, KV pool, recurrent states, compute buffers
    and the MTP draft's own. qwen4exp only (the constants above are its); None for anything else."""
    meta = metadata(Path(model['path']))
    arch = meta.get('general.architecture')
    if arch != 'qwen4exp': return None
    cells, slots = kv_pool_cells(cfg), cfg.get('parallel', 1)
    n_attn = meta[f'{arch}.block_count'] // max(1, meta.get(f'{arch}.full_attention_interval', 1))
    kvh, kl = meta[f'{arch}.attention.head_count_kv'], meta[f'{arch}.attention.key_length']
    vl, il = meta.get(f'{arch}.attention.value_length', kl), meta.get(f'{arch}.attention.indexer.key_length', 0)
    elem = KV_ELEM_BYTES.get(cfg['kv'], 2.0)
    per_cell_layer = kvh * (kl + vl) * elem + (il * 2 if cfg.get('qsa') else 0)
    need = sum(v for k, v in tb.items() if k not in CPU_TENSORS) * 1.01        # rows padded on the GPU
    need += cells * n_attn * per_cell_layer + slots * RS_BYTES_PER_SLOT
    need += cfg['ubatch'] * COMPUTE_BYTES_PER_UBATCH_TOKEN + cells * COMPUTE_BYTES_PER_CELL
    if cfg.get('mtp') and cfg.get('draft') and Path(cfg['draft']).is_file():
        # the draft's K/V are f16 whatever the target's type
        need += Path(cfg['draft']).stat().st_size * 1.1 + cells * (kvh * (kl + vl) * 2 + (il * 2 if cfg.get('qsa') else 0)) + 300 * MIB
        need += cells * DRAFT_MASK_BYTES_PER_CELL * draft_ubatch(cells) / 2048
    return need + HOST_EXPERT_MARGIN


def total_ram_bytes():
    if os.name != 'nt': return None
    ms = _MemoryStatus(); ms.dwLength = ctypes.sizeof(ms)
    return ms.ullTotalPhys if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None


def host_expert_layers(model, cfg, dedicated=None, ram=None):
    """The layers whose experts this load keeps in system memory: none when it fits the carve, else the last layers'
    until the rest does. Bounded by system memory, leaving 16 GiB of it to Windows and everything else."""
    if os.environ.get('STRIX_HOST_EXPERTS') == '0': return []      # leave the overflow to the display driver
    dedicated = dedicated_vram_bytes() if dedicated is None else dedicated
    if not dedicated: return []
    try:
        tb = tensor_bytes(model_shards(model))
        need = gpu_need_bytes(model, cfg, tb)
    except (OSError, ValueError, KeyError, struct.error):
        return []
    dedicated = dedicated * CARVE_USABLE
    if need is None or need <= dedicated: return []
    ram = total_ram_bytes() if ram is None else ram
    cap = max(0, (ram or 0) - (16 << 30)) if ram else float('inf')
    per_layer = {}
    for name, size in tb.items():
        m = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$', name)
        if m: per_layer[int(m[1])] = per_layer.get(int(m[1]), 0) + size
    moved, out = 0, []
    for layer in sorted(per_layer, reverse=True):
        if need - moved <= dedicated or moved + per_layer[layer] > cap: break
        out.append(layer); moved += per_layer[layer]
    return sorted(out)


# How the runtime says it ran out of device memory: ggml's allocator ("cudaMalloc failed: out of
# memory"), the KV cache and graph reserves ("failed to allocate ... buffer"), and HIP's own name.
OOM_SIGNS = ('out of memory', 'cudamalloc failed', 'hiperroroutofmemory', 'failed to allocate')


def exit_reason(log):
    """Why a load ended, from the tail of its log: ('oom', line) when it ran out of memory, ('error',
    line) for the last line that looks like one, ('exited', '') when the log says nothing."""
    try:
        path = Path(log)
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 65536))
            tail = f.read().decode('utf-8', 'replace')
    except OSError:
        return 'exited', ''
    lines = [l.strip() for l in reversed(tail.splitlines()) if l.strip()]
    oom = next((l for l in lines if any(s in l.lower() for s in OOM_SIGNS)), None)
    if oom: return 'oom', oom
    err = next((l for l in lines if re.search(r'\berror\b|failed|abort|exception|\bE\b', l, re.I)), None)
    return ('error', err) if err else ('exited', '')


# When Windows' commit (RAM + page file) runs out under a loaded server, a decode step fails with
# "bad allocation" and every answer in flight is stopped with an error; an optional copy - checkpoints
# handed to the disk tier - is skipped instead ("not enough memory"). The chat shows only an answer that ends
# mid-sentence, so status() reports both from the log, and the pages say what happened and what to change.
def memory_events(log):
    """Answers stopped and saves skipped for lack of memory in a server's log, with the wall-clock time of the
    last of each; None when there were none. Log lines carry minutes.seconds.ms since the launch, which the
    log's own name records."""
    try:
        path = Path(log)
        data = path.read_bytes()
    except (OSError, TypeError):
        return None
    if b'bad allocation' not in data and b'not enough memory' not in data:
        return None
    m = re.search(r'jan-managed-(\d{8}-\d{6})', path.name)
    start = dt.datetime.strptime(m.group(1), '%Y%m%d-%H%M%S') if m else None

    def when(line):
        t = re.match(r'(\d+)\.(\d{2})\.(\d{3})', line)
        if not (start and t): return None
        return (start + dt.timedelta(minutes=int(t[1]), seconds=int(t[2]), milliseconds=int(t[3]))).isoformat(timespec='seconds')

    out = {'answers_stopped': 0, 'saves_skipped': 0}
    for line in data.decode('utf-8', 'replace').splitlines():
        if 'send_error' in line and 'bad allocation' in line:
            out['answers_stopped'] += 1
            out['last_stop'] = when(line)
        elif 'not enough memory' in line:
            out['saves_skipped'] += 1
            out['last_skip'] = when(line)
    return out if out['answers_stopped'] or out['saves_skipped'] else None


def runtime_environment(cfg):
    # start from a clean slate: a stray LLAMA_*/GGML_*/STRIX_* from a shell would silently change
    # the graph, and an inherited value is never what the profile asked for
    env = {k: v for k, v in os.environ.copy().items()
           if not k.upper().startswith(('LLAMA_', 'GGML_', 'STRIX_'))}
    env.update({k: str(v) for k, v in HIP_GATES.items()})
    on = cfg.get('qsa', False)
    env.update({k: ('1' if on else '0') for k in HIP_QSA_GATES})
    env['LLAMA_QSA_QUERY_STRIP'] = '512' if on else '0'
    # No *_ENABLE_UNIFIED_MEMORY: up to 0.1.14 this set GGML_HIP_ENABLE_UNIFIED_MEMORY, which nothing reads - ggml
    # checks only GGML_CUDA_ENABLE_UNIFIED_MEMORY, and only for being set (so "0" would turn it on) - and the clean
    # slate above drops both from the inherited environment. Allocations go to the carve, and to shared GPU
    # memory when the display driver puts them there.
    # Q6_K decode twins of the Q8_0 trunk (llama-model.cpp build_decode_twins): batches of <= 8 tokens
    # read 23% fewer trunk bytes; prefill keeps the Q8_0 originals. No UI control - measured
    # prefill-neutral and 4% on decode for 2.9 GB, and at ctx 262144 it can stop a long prompt loading.
    env['LLAMA_TRUNK_DECODE_Q6K'] = '1' if cfg.get('trunk_decode_q6k', False) else '0'
    # the server's prompt cache gets a disk tier (see DEFAULTS); the directory lives with the settings
    if cfg.get('prompt_cache_disk', False):
        env['STRIX_PROMPT_CACHE_DIR'] = str(DATA / 'prompt-cache')
        env['STRIX_PROMPT_CACHE_MIB'] = str(cfg.get('prompt_cache_disk_mib') or PROMPT_CACHE_DISK_MIB)
        env['STRIX_PROMPT_CACHE_BLOCK'] = str(PROMPT_CACHE_BLOCK_TOKENS)
    # A draft pays for itself on one conversation, less on several at once: each drafted token is verified, and
    # in this mixture of experts a verified token reads ~10 more experts' weights, which conversations decoding
    # together cannot share. So the server drafts draft_max tokens for one generating slot, at most 2 for two to
    # four, none from five on (STRIX_SPEC_DRAFT_BY_SLOTS). Measured 2026-09-26, tok/s summed: four conversations
    # 55.7 without drafts -> 62.6 with 2 at ~4K tokens each, 50.6 -> 56.1 at ~20K; six 64.4 without, 60.7 with 2;
    # eight 70.7 without, 64.0 with 2 (greedy). Before 0.2.3's small-batch router and expert kernels four had been
    # better without.
    if cfg.get('mtp') and cfg.get('parallel', 1) > 1:
        dm = int(cfg['draft_max'])
        env['STRIX_SPEC_DRAFT_BY_SLOTS'] = ','.join(str(x) for x in (dm, min(dm, 2), min(dm, 2), min(dm, 2), 0))
    # A request that samples (temperature above 0) has its drafts drawn and verified by speculative sampling since 0.4.1
    # and accepted less often than a greedy one's - at temperature 0.7 on prose ~67% at the first position, ~86% greedy -
    # so a draft position pays less, and it has its own table, also with one slot (STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED).
    # Measured 2026-10-01 on Chinese prose at Jan's sampling (temperature 0.7, top_k 20, top_p 0.8), tok/s summed:
    # one conversation 36.3 with 2 drafts against 35.5 with 3 (three seed sets); two 55.6 with 2, 54.6 with 1; three
    # 67.6 with 1 against 63.5 with 2; four 77.0 with 1 against 71.2 with 2 and 70.7 without; six 87.0 with 1 against
    # 89.2 without.
    if cfg.get('mtp'):
        dm = int(cfg['draft_max'])
        env['STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED'] = ','.join(str(x) for x in (min(dm, 2), min(dm, 2), min(dm, 1), min(dm, 1), 0))
    # Several conversations decoding together: from seven tokens a step the routed experts leave the vector kernel's
    # single pass - for chunks of it up to 16 tokens since 0.2.3, for the tiled kernel past that, which dequantizes an
    # expert once for all its tokens (eight conversations +6% summed, measured before the chunks). Not with one slot,
    # where the limit stays upstream's and the results stay those of earlier versions.
    if cfg.get('parallel', 1) > 1:
        env['STRIX_MOE_VEC_MAX'] = '6'
    # The draft context reserves a dense mask over the whole KV pool for its ubatch, ~4.8 KiB a cell at the 2048 above:
    # 2.4 GB at 512K cells, and at 768K the load died on the draft's 3.9 GB compute buffer with the target already in
    # (a user's log, 2026-09-26). Past 512K cells the draft takes a smaller ubatch, so its reserve stays at the 512K
    # figure; only the draft's share of a prefill (its one layer) runs in more, smaller passes.
    if cfg.get('mtp'):
        env['STRIX_SPEC_DRAFT_UBATCH'] = str(draft_ubatch(kv_pool_cells(cfg)))
    if not bundled_rocm():
        env['PATH'] = str(ROCM_BIN) + os.pathsep + os.environ.get('PATH', '')
    return env


def api_model_name(model):
    """The name /v1/models lists for the loaded model, the one API clients send as "model": the file's name without
    the shard suffix, e.g. Qwen3.8-Flash-Next-UD-IQ4_XS. Without --alias llama-server lists the full path. With one
    model loaded it answers whatever name a request carries, but some clients check the name against that list
    first (issue #9). --alias splits its value at commas."""
    return re.sub(r'(-\d{5}-of-\d{5})?\.gguf$', '', Path(model['path']).name, flags=re.I).replace(',', '_')


def argv(model, cfg, net=None):
    net = net or network()
    pool = kv_pool_cells(cfg)
    args = [str(selected_runtime(cfg)), '-m', model['path'], '-ngl', str(cfg['gpu_layers']), '-c', str(pool),
            '-b', str(cfg['batch']), '-ub', str(cfg['ubatch']), '-t', str(cfg['threads']), '--poll', '0',
            '--fit', 'off', '-np', str(cfg.get('parallel', 1)), '-fa', cfg['flash_attention'], '-ctk', cfg['kv'], '-ctv', cfg['kv'], '--jinja',
            '--host', net['host'], '--port', str(net['port']), '--alias', api_model_name(model)]
    if net['api_key']:
        # every endpoint but /health then wants it: the manager's own requests and the app's chat send it
        args += ['--api-key', net['api_key']]
    if cfg.get('parallel', 1) > 1:
        # without it the pool is split evenly and each slot would see context/parallel tokens; -kvu keeps one
        # shared pool so a single conversation can still use the whole context when the others are idle
        args += ['-kvu']
        if pool > cfg['context']:
            # a pool larger than one conversation (kv_pool): each conversation is still capped at the context
            args += ['--kv-unified-per-slot', str(cfg['context'])]
    think = cfg['thinking'] if not isinstance(cfg['thinking'], bool) else ('high' if cfg['thinking'] else 'off')
    kwargs = ({'enable_thinking': False} if think == 'off'
              else {'enable_thinking': True, 'reasoning_effort': THINKING[think]})
    # --no-cache-idle-slots: upstream saves AND clears every idle slot on each new task when the KV is unified
    # (-kvu, i.e. more than one slot), so talking to A, then B, then A read A back from disk and wrote B out,
    # ~6 s a switch for long conversations, although the cells were already allocated. Without it they stay
    # in their slots until the pool is actually full, and the disk tier, when it is on, writes their rows as they
    # are computed and their state when they leave.
    args += ['--cache-prompt', '--cache-ram', str(PROMPT_CACHE_RAM_MIB), '--no-cache-idle-slots',
             '--ctx-checkpoints', str(CTX_CHECKPOINTS), '--checkpoint-min-step', str(CHECKPOINT_MIN_STEP),
             '--chat-template-kwargs', json.dumps(kwargs, separators=(',',':'))]
    if cfg.get('vision', False):
        args += ['--mmproj', str(mmproj_path(cfg, model))]
    host = host_expert_layers(model, cfg)
    if host:
        args += ['-ot', r'blk\.(%s)\.ffn_(gate|up|down)_exps\.weight=ROCm_Host' % '|'.join(map(str, host))]
    # this runtime reads the per-layer embedding table itself with offset I/O, so it must not be
    # pinned to CPU memory
    args += ['--load-mode', 'none', '--lazy-mode', 'on-direct']
    spec_types = []
    if cfg.get('ngram_spec', False):
        spec_types.append('ngram-mod')
        # Bound recurrent rollback storage separately from the MTP draft length.
        args += ['--spec-ngram-mod-n-match', '24', '--spec-ngram-mod-n-min', '4', '--spec-ngram-mod-n-max', '8']
    if cfg['mtp']:
        spec_types.append('draft-mtp')
        # With experts in system memory (host above) one draft at most. A verify of 1 + n drafts reads n more tokens'
        # experts there, and from two drafts on the GPU stalls ~400 ms every ~5.3 s of wall clock (since 0.4.7; gone
        # with the experts in the carve or with one draft). Measured 2026-10-08, 64 GB carve, MTP on, greedy: 86K decode
        # 27.1-27.7 ms/token with three drafts (stalls included) against 26.6-26.7 with one; 1500 tokens of short
        # context 42.4 s against 40.3 s. The per-slot tables below only lower this further.
        draft_max = 1 if host else cfg['draft_max']
        args += ['-md', cfg['draft'], '-ngld', str(cfg['gpu_layers']), '--spec-draft-n-max', str(draft_max), '--spec-draft-p-min', str(cfg['draft_min'])]
    if spec_types: args += ['--spec-type', ','.join(spec_types)]
    return args


def masked(args):
    """A command with the API key's value replaced, for what is kept and shown: process.json, the Logs page."""
    return [('***' if i and args[i - 1] == '--api-key' else a) for i, a in enumerate(args)]


def masked_command(cmd):
    return re.sub(r'(--api-key\s+)(?:"[^"]*"|\S+)', r'\g<1>***', cmd)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + \
               [(name, ctypes.c_ulonglong) for name in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                                                         'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]


def commit_bytes():
    """Windows' commit limit (RAM + page file) and what is left of it, or None where it cannot be read.

    On this machine the GPU's large allocations count against it one for one - the weights, the target's K/V,
    the compute buffers - and so do the context checkpoints a long conversation keeps in RAM (~113 MiB every
    ~8K tokens, up to 32 a slot). When it runs out, whatever allocates next fails and the server reports
    "bad allocation", however much GPU memory is free (docs/measuring.md)."""
    if os.name != 'nt': return None
    ms = _MemoryStatus(); ms.dwLength = ctypes.sizeof(ms)
    if not ctypes.WinDLL('kernel32').GlobalMemoryStatusEx(ctypes.byref(ms)): return None
    return ms.ullTotalPageFile, ms.ullAvailPageFile


def process_identity(pid, terminate=False, expected=None):
    """Hold the same process handle while checking identity and terminating it."""
    if os.name != 'nt': fail('windows_only')
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    k.TerminateProcess.argtypes = [wintypes.HANDLE,wintypes.UINT]
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    handle = k.OpenProcess(0x1000 | 0x100000 | (1 if terminate else 0), False, int(pid))
    if not handle: return None
    try:
        buf, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not k.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)): return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]): return None
        birth = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
        result = {'pid':int(pid), 'exe':str(Path(buf.value).resolve()), 'birth':birth}
        if terminate:
            # the identity (pid, executable, birth time) is what was adopted or started; that it
            # matches, and that it is a llama-server at all, is the check - so a server another
            # copy of this manager started can be unloaded too, and nothing else ever is
            if result != expected or Path(result['exe']).name.lower() != 'llama-server.exe':
                fail('identity_changed')
            if not k.TerminateProcess(handle,0): fail('stop_failed')
            k.WaitForSingleObject(handle,10000)
        return result
    finally: k.CloseHandle(handle)


def listening(s):
    """The address and API key of the server state() found, as its launch or adoption recorded them: the manager's
    own requests go there, and the pages show it, until it stops - a network setting saved meanwhile applies to the
    next load. A server an earlier version started has only its command line, which named the port, and no key.
    Without a server, the next load's: network()."""
    if not s.get('identity'):
        return network()
    cmd = s.get('command') or ''
    port, host = s.get('port'), s.get('host')
    if type(port) is not int:
        found = re.search(r'--port\s+(\d+)', cmd)
        port = int(found.group(1)) if found else network()['port']
    if not isinstance(host, str):
        found = re.search(r'--host\s+(\S+)', cmd)
        host = found.group(1) if found else LOOPBACK
    return {'host': host, 'port': port, 'api_key': s.get('api_key') or ''}


def base_url(server):
    # a server on 0.0.0.0 listens on loopback too, and this PC reaches it there
    return f'http://{LOOPBACK if server["host"] == ANY_ADDRESS else server["host"]}:{server["port"]}'


def auth_headers(server):
    # with a key, llama-server answers only /health without it
    return {'Authorization': 'Bearer ' + server['api_key']} if server.get('api_key') else {}


def public(s):
    """A process record without the API key it keeps for the manager's own requests: what status() and start hand
    to the pages, which never get the key that way."""
    return {k: v for k, v in s.items() if k != 'api_key'}


def lan_addresses():
    """This PC's IPv4 addresses other devices may reach it at, best effort: the one the default route leaves by
    first, then the others its name resolves to, without loopback and link-local (169.254.x) ones. The UDP connect
    sends nothing; it only picks the route."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))   # TEST-NET-1 (RFC 5737): reserved, never routed anywhere real
            found.append(probe.getsockname()[0])
    except OSError: pass
    try: found += [info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError: pass
    return [a for a in dict.fromkeys(found) if isinstance(a, str) and not a.startswith(('127.', '169.254.', '0.'))]


def lan_endpoints(server):
    """The endpoints other devices use for a server bound to server['host']: one per lan_addresses() for 0.0.0.0,
    the address itself for another one, none for loopback."""
    try:
        if ipaddress.IPv4Address(server['host']).is_loopback: return []
    except ValueError:
        return []
    addresses = lan_addresses() if server['host'] == ANY_ADDRESS else [server['host']]
    return [f'http://{a}:{server["port"]}/v1' for a in addresses]


def network_view(s=None):
    """Configuration › Network: settings.json's values (`saved`, what the page edits and save_network takes); those
    in effect for the next load, RULITH_* applied, and which fields a variable forces; this PC's addresses, for the
    endpoints other devices use; and in `server` the endpoint and key a client of this PC uses now - the running
    server's, else the next load's - which the app's chat follows. The one answer that carries a key."""
    s = state() if s is None else s
    net, srv = network(), listening(s)
    return {'saved': saved_network(), **{k: net[k] for k in ('port', 'lan', 'host', 'api_key', 'forced')},
            'addresses': lan_addresses(), 'server': {'endpoint': base_url(srv) + '/v1', 'api_key': srv['api_key']}}


def persist_conversations(saved):
    """Before a stop, with the disk tier on: the conversations still in the server's slots leave memory with it,
    and the tier writes a conversation's recurrent state only when it leaves (POST /strix/persist answers once
    the writer is done). Best effort - a runtime without the endpoint, or one that does not answer in time, is
    stopped all the same."""
    if not (saved.get('profile') or {}).get('prompt_cache_disk'): return
    try:
        srv = listening(saved)
        req = urllib.request.Request(base_url(srv) + '/strix/persist', data=b'{}', method='POST',
                                     headers={'Content-Type': 'application/json', **auth_headers(srv)})
        with HTTP.open(req, timeout=150) as res: res.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError): pass


def discover():
    script = "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process -Filter \"Name = 'llama-server.exe'\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    p = subprocess.run(['powershell.exe','-NoProfile','-Command',script],capture_output=True,encoding='utf-8',errors='replace',creationflags=HIDDEN,timeout=15)
    entries = json.loads(p.stdout) if p.stdout.strip() else []
    if isinstance(entries,dict): entries=[entries]
    return entries


def state():
    saved = read_json(DATA / 'process.json', {})
    ident = saved.get('identity')
    if ident and managed_runtime(ident.get('exe')) and process_identity(ident['pid']) == ident: return saved
    if ident and saved.get('adopted') is False and saved.get('log'):
        # A process this manager started is gone without a stop. Record why, once, so the page can
        # say what happened instead of silently going back to "not loaded".
        reason, line = exit_reason(saved['log'])
        saved = {'last_log': saved['log'], 'exited': dict(reason=reason, line=line, model_id=saved.get('model_id'),
                 profile=saved.get('profile'), log=saved['log'])}
        atomic_json(DATA / 'process.json', saved)
        return saved
    # Adopt a server of ours on the configured local endpoint: the exact project binary, or a
    # llama-server on our port started with our launch flags by another copy of this manager (an
    # earlier install, a checkout in another directory). Without the second case a model loaded
    # from one copy could not be unloaded from the next, and its port stayed taken. The port is the
    # configured one (8080 unless Configuration › Network or RULITH_PORT says otherwise), on loopback
    # or on the network: an earlier version's server is on 127.0.0.1:8080, the defaults.
    net = network()
    hosts = {LOOPBACK, ANY_ADDRESS, net['host']}
    for p in discover():
        cmd = p.get('CommandLine') or ''
        ours = managed_runtime(p.get('ExecutablePath')) or (
            Path(p.get('ExecutablePath') or '').name.lower() == 'llama-server.exe' and '--lazy-mode on-direct' in cmd)
        host = re.search(r'--host\s+(\S+)', cmd)
        if ours and re.search(rf'--port\s+{net["port"]}(?:\s|$)', cmd) and host and host.group(1) in hosts:
            model_match = re.search(r'(?:^|\s)-m\s+(?:"([^"]+)"|(\S+))', cmd)
            log_match = re.search(r'--log-file\s+(?:"([^"]+)"|(\S+))', cmd)
            key_match = re.search(r'--api-key\s+(?:"([^"]*)"|(\S+))', cmd)
            identity = process_identity(p['ProcessId'])
            if identity and Path(identity['exe']) == Path(p['ExecutablePath']).resolve():
                # its key is kept for the manager's own requests, and only there: the command shown has it masked
                saved = dict(identity=identity, model_path=next((v for v in model_match.groups() if v), '') if model_match else '',
                             log=next((v for v in log_match.groups() if v), '') if log_match else '',
                             command=masked_command(cmd), adopted=True, host=host.group(1), port=net['port'],
                             api_key=next((v for v in key_match.groups() if v), '') if key_match else '')
                atomic_json(DATA/'process.json',saved)
                return saved
    result = {'last_log':saved.get('log', saved.get('last_log',''))}
    if saved.get('exited'): result['exited'] = saved['exited']
    return result


def http_json(path, server):
    """GET one of the server's JSON endpoints at `server` (listening(), network()), with its key."""
    req = urllib.request.Request(base_url(server) + path, headers=auth_headers(server))
    with HTTP.open(req, timeout=1.5) as res: return json.load(res)


def status():
    s = state()
    # the endpoint is the running server's, else the one the next load takes; never the key, only whether there is one
    srv = listening(s)
    result = {**public(s), 'status':'stopped', 'endpoint':base_url(srv)+'/v1', 'api_key_set':bool(srv['api_key']),
              'runtime':s.get('identity', {}).get('exe', str(RUNTIME)),
              'runtime_available': runtime_available(), 'runtime_info': runtime_info(),
              'dedicated_vram': dedicated_vram_bytes()}
    lan = lan_endpoints(srv)
    if lan: result['lan_endpoints'] = lan
    e = s.get('exited') or {}
    if e.get('reason') in ('oom', 'error'):
        # a plain exit with nothing in the log is not reported: the app closing takes the server
        # with it, and "the last load failed" would be the wrong thing to say about that
        result['failure'] = e.get('line') or FAILURES[e['reason']]
        # the code only when the text is ours to translate, not a line quoted from the log
        result['failure_code'] = None if e.get('line') else e['reason']
    if s.get('identity'):
        result['status']='loading'
        result['model_name'] = next((x.get('name') for x in catalog()['models'] if x.get('path') == s.get('model_path')), None) or Path(s.get('model_path', '')).stem
        # a network setting saved while it runs applies when it loads again; the pages say so meanwhile
        net = network()
        result['network_pending'] = (srv['host'], srv['port'], srv['api_key']) != (net['host'], net['port'], net['api_key'])
        try:
            if http_json('/health', srv).get('status')=='ok':
                result['status']='ready'
                models=http_json('/v1/models', srv)['data']
                result['served_models']=models
        except Exception: pass
        # a loaded server with little commit left fails its next allocation with "bad allocation"; say so
        # while there is still time to shrink the pool or enlarge the page file
        commit = commit_bytes()
        if commit and result['status'] == 'ready':
            result['commit_limit'], result['commit_available'] = commit
            result['commit_low'] = commit[1] < COMMIT_LOW_BYTES
        events = memory_events(s.get('log'))
        if events:
            result['memory_events'] = events
    return result


def slots():
    """What each of the running server's slots is doing, for the Logs page: from llama-server's /slots, which
    answers with counters only (no prompt text unless LLAMA_SERVER_SLOTS_DEBUG is set), and the requests waiting in its
    queue. A server busy with a large batch answers late; None then, and the page keeps what it showed."""
    s = state()
    if not s.get('identity'):
        return {'slots': []}
    srv = listening(s)
    try:
        with HTTP.open(urllib.request.Request(base_url(srv) + '/slots', headers=auth_headers(srv)), timeout=2) as res:
            data = json.load(res)
            # requests waiting for a slot or for room in the KV pool (since 0.3.3; older runtimes do not say)
            waiting = int(res.headers.get('X-Strix-Waiting') or 0)
    except Exception:
        return {'slots': None}
    out = []
    for x in data if isinstance(data, list) else []:
        nt = (x.get('next_token') or [{}])[0]
        out.append({'id': x.get('id'), 'active': bool(x.get('is_processing')), 'task': x.get('id_task'),
                    'context': x.get('n_prompt_tokens', 0), 'prompt_processed': x.get('n_prompt_tokens_processed', 0),
                    'prompt_cached': x.get('n_prompt_tokens_cache', 0), 'generated': nt.get('n_decoded', 0)})
    return {'slots': out, 'waiting': waiting}


def logs(offset=0):
    s = state()
    path = Path(s.get('log') or s.get('last_log') or ROOT/'logs'/'not-started.log')
    if not path.resolve().is_relative_to((ROOT/'logs').resolve()): fail('log_path')
    if not path.exists(): return {'text':'','offset':0,'file':str(path),'reset':False}
    size = path.stat().st_size
    offset = max(0,int(offset))
    reset = offset > size
    if reset: offset=0
    if offset == 0: offset=max(0,size-65536)
    with path.open('rb') as f:
        f.seek(offset)
        b=f.read(65536)
        # Do not split a UTF-8 sequence across polling boundaries.
        tail=0
        for n in range(0,4):
            try: text=b[:len(b)-n if n else None].decode('utf-8'); tail=n; break
            except UnicodeDecodeError:
                if n==3: text=b.decode('utf-8','replace')
        return {'text':text,'offset':offset+len(b)-tail,'file':str(path),'reset':reset}


def launch(m, cfg):
    """Start the runtime for model m with the already validated profile cfg: check the port, write
    the log banner, record the process identity."""
    net = network()
    try: http_json('/health', net); fail('port_busy', port=net['port'])
    except (urllib.error.URLError,TimeoutError): pass
    # Check port before allocating model memory; do not stop unrelated engines. A server on 0.0.0.0
    # also answers on loopback, where the manager and the chat reach it, so both must be free.
    for host in (net['host'], LOOPBACK) if net['host'] == ANY_ADDRESS else (net['host'],):
        with socket.socket() as sock:
            try: sock.bind((host, net['port']))
            except OSError as e:
                # RULITH_HOST naming an address this PC does not have (WSAEADDRNOTAVAIL) is not a busy port
                if e.errno in (errno.EADDRNOTAVAIL, 10049): fail('host_unavailable', host=host)
                fail('port_busy', port=net['port'])
    runtime = selected_runtime(cfg)
    if not runtime.is_file() or not (runtime.parent/'ggml-hip.dll').is_file():
        fail('runtime_missing', name=runtime.parent.name)
    log=ROOT/'logs'/('jan-managed-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.log')
    log.parent.mkdir(exist_ok=True)
    command=argv(m,cfg,net)
    env=runtime_environment(cfg)
    # the server's request log beside its console log: every request as received, the raw output and the parsed tool
    # calls, for debugging (STRIX_REQUEST_LOG, since 0.5.2; repeated long parts are written once per file).
    # RULITH_REQUEST_LOG=0 in the environment turns it off
    if os.environ.get('RULITH_REQUEST_LOG') != '0':
        env['STRIX_REQUEST_LOG']=str(log.with_suffix('.requests.jsonl'))
    banner = (f'[strixllama] runtime={runtime.parent.name} (HIP/ROCm); LLAMA_MMB_HC16={env["LLAMA_MMB_HC16"]} '
              f'(must stay 0 on Windows); gates={sum(1 for k in env if k.startswith("LLAMA_"))}; '
              f'QSA={"on" if cfg["qsa"] else "off"} (this runtime has no context threshold); '
              f'MTP={"on" if cfg["mtp"] else "off"}'
              f'{f" (draft ubatch capped to {env['STRIX_SPEC_DRAFT_UBATCH']})" if cfg["mtp"] else ""}; '
              f'n-gram draft={"on (match=24, min=4, max=8)" if cfg["ngram_spec"] else "off"}; '
              f'vision={"on" if cfg["vision"] else "off"}; '
              f'PLE reader=on-direct; rocm={"bundled beside the server" if bundled_rocm() else ROCM_BIN}; '
              f'listen={net["host"]}:{net["port"]}{" (API key required)" if net["api_key"] else ""}\n')
    with log.open('wb') as f:
        f.write(banner.encode('utf-8'))
        f.flush()
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=HIDDEN,cwd=ROOT)
    ident=process_identity(proc.pid)
    if not ident: fail('launch_failed')
    # the address and key it listens with stay recorded while it runs, for the manager's own requests: a network
    # setting saved meanwhile is for the next load. The key itself only here, never in the command kept and shown.
    saved=dict(identity=ident,model_id=m['id'],model_path=m['path'],log=str(log),command=subprocess.list2cmdline(masked(command)),profile=cfg,
               started_at=dt.datetime.now().astimezone().isoformat(),adopted=False,
               runtime_env={k:v for k,v in env.items() if k.startswith(('LLAMA_','GGML_','STRIX_'))},
               host=net['host'],port=net['port'],api_key=net['api_key'])
    atomic_json(DATA/'process.json',saved)
    return public(saved)


def handle(op, data):
    if op=='catalog': return catalog(bool(data.get('refresh')))
    if op=='status': return status()
    if op=='logs': return logs(data.get('offset',0))
    if op=='slots': return slots()
    if op=='profile':
        m=model_by_id(data['id'])
        # what the companion switches can be turned on with: the page explains an off switch by it
        companions={'draft':family_draft(),'mmproj':mmproj_path({},m).is_file(),
                    'draft_head':any('-head-' in Path(p).name.lower() for p in [family_draft() or ''])}
        return {'model':m,'profile':profile(m),'companions':companions}
    if op=='roots':
        roots=data['roots']
        if not isinstance(roots,list) or not 1 <= len(roots) <= 12: fail('roots_count')
        parsed=[str(Path(r).resolve(strict=True)) for r in roots]
        if any(not Path(r).is_dir() for r in parsed): fail('root_missing')
        cfg=settings();cfg['roots']=list(dict.fromkeys(parsed));atomic_json(DATA/'settings.json',cfg)
        return catalog(True)
    if op=='save':
        m=model_by_id(data['id']); cfg=validate_profile(data['profile'],m)
        all_cfg=settings();all_cfg['profiles'][m['id']]=cfg;atomic_json(DATA/'settings.json',all_cfg)
        return {'profile':cfg,'restart_required':bool(state().get('identity')),'argv':masked(argv(m,cfg))}
    if op=='network': return network_view()
    if op=='save_network':
        # settings.json's own values; a RULITH_* variable still overrides what it sets
        net=check_network(data)
        all_cfg=settings();all_cfg['network']=net;atomic_json(DATA/'settings.json',all_cfg)
        s=state()
        return {'network':network_view(s),'restart_required':bool(s.get('identity'))}
    if op=='stop':
        s=state()
        if s.get('identity'):
            persist_conversations(s)
            process_identity(s['identity']['pid'],True,s['identity'])
        atomic_json(DATA/'process.json',{'last_log':s.get('log',s.get('last_log',''))})
        return {'status':'stopped'}
    if op=='start':
        m=model_by_id(data['id'])
        if m['role']!='model': fail('not_a_model')
        cfg=validate_profile(data.get('profile',profile(m)),m)
        if state().get('identity'): fail('already_loaded')
        return {**launch(m,cfg),'status':'loading'}
    fail('unknown_op')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536: fail('request_too_large')
        request=json.loads(raw)
        if not isinstance(request,dict): fail('request_invalid')
        # Serialize all access across the short-lived native IPC helpers.
        DATA.mkdir(parents=True,exist_ok=True)
        with (DATA/'manager.lock').open('a+b') as lock:
            import msvcrt
            if lock.tell()==0: lock.write(b'0');lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
            try: result=handle(request['op'],request.get('data',{}))
            finally: lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        print(json.dumps({'ok':True,'data':result},ensure_ascii=False))
    except ManagerError as exc:
        print(json.dumps({'ok':False,'error':str(exc),'code':exc.code,'params':exc.params},ensure_ascii=False))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False))
        sys.exit(1)


if __name__=='__main__': main()
