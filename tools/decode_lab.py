#!/usr/bin/env python3
"""Run decode probes against the production runtime under a named set of environment overrides.

Each config launches llama-server directly (manager.argv + manager.runtime_environment, so every
flag matches production), waits for health, walks growing context depths with cache_prompt on, and
reports ms/token per depth plus the speculation counters the server returns.

    python tools/decode_lab.py --config base
    python tools/decode_lab.py --config graphtime --config nodetime --n 8
    python tools/decode_lab.py --config base --set mtp=true ngram_spec=true --name spec-on

Configs are declared in CONFIGS below: env overrides plus optional profile overrides. --keep skips
the launch and probes whatever server is already listening, which is how to measure production
itself without disturbing it.
"""
import argparse
import json
import random
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def default_model_id():
    """The model to probe: whatever the manager's catalog lists first, unless --model says otherwise.

    A model id is a hash of the file's location, so it is specific to one machine and cannot be a
    constant in the source. The first model is not necessarily the one under study - with other
    models on disk it is whichever sorts first - so pass --model and read the "model:" line.
    """
    found = call("catalog", {}).get("data", {}).get("models", [])
    # role, because the catalog also lists draft heads and vision projectors and sorts them first
    models = [m for m in found if m.get("role") == "model" and not (m.get("error") or m.get("missing"))]
    if not models:
        sys.exit("no usable model in the catalog - run the manager's scan first, or pass --model <id>")
    return models[0]["id"]

# env overrides per named config; profile overrides go in PROFILE_OVERRIDES
CONFIGS = {
    # production as shipped
    "base": {},
    # per-graph host/submit/gpu split, and whether HIP graphs engage at decode (graph=1)
    "graphtime": {"LLAMA_GRAPH_TIMING": "1"},
    # per-dispatch HIP events; needs graphs off, so it is ~3x slower and only the shape matters
    "nodetime": {"STRIX_NODE_TIMING": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"},
    # graphs off with no instrumentation: the reference the nodetime numbers must be read against
    "nograph": {"GGML_CUDA_DISABLE_GRAPHS": "1"},
    # the long-context path, piece by piece, with graphs left on, as production runs them
    "skip-fa": {"STRIX_SKIP_OPS": "FLASH_ATTN_EXT"},
    "skip-idx": {"STRIX_SKIP_OPS": "LIGHTNING_INDEXER"},
    "skip-topk": {"STRIX_SKIP_OPS": "TOP_K,ARGSORT"},
    "skip-sel": {"STRIX_SKIP_OPS": "FLASH_ATTN_EXT,LIGHTNING_INDEXER,TOP_K,ARGSORT"},
    # the decode-side gather, off: falls back to dense attention over the whole cache
    "gather-off": {"LLAMA_QSA_DECODE_GATHER": "0"},
    # the block-key cache, off: every graph rebuilds every block key
    "kb-off": {"LLAMA_QSA_BLOCK_KEY_CACHE": "0"},
    # CPU-side split of each speculative pass: draft / target decode / catch-up / sampling / rest
    "spectime": {"STRIX_SPEC_TIMING": "1"},
    # and inside one llama_decode: prep / graph / set_inputs / backend / readback
    "dectime": {"STRIX_SPEC_TIMING": "1", "STRIX_DECODE_TIMING": "1"},
    # HIP graphs keyed on the first node only (stock): shapes alternating on one context reset each other
    "keyfirst": {"STRIX_GRAPH_KEY_SHAPE": "0"},
    # the draft head's decode steps take sparse attention too, not just its prefill
    "dft-qsa": {"LLAMA_MTP_QSA_MIN_T": "1"},
}

PROFILE_OVERRIDES = {}


def call(op, data):
    p = subprocess.run([sys.executable, str(ROOT / "tools/manager.py")], input=json.dumps({"op": op, "data": data}),
                       capture_output=True, text=True, encoding="utf-8")
    return json.loads(p.stdout)


def parse(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def wait_ready(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def wait_vram_free(limit_gb=6.0, timeout=180):
    """The production server holds the whole carve; a second one cannot allocate until it is gone."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)."
                            "CounterSamples | Measure-Object -Property CookedValue -Sum | "
                            "%{[math]::Round($_.Sum/1GB,2)}"],
                           capture_output=True, text=True)
        try:
            if float(p.stdout.strip()) < limit_gb:
                return True
        except ValueError:
            return True
        time.sleep(5)
    return False


def probe(port, steps, n_predict, words):
    rows = []
    for n_words in steps:
        body = {"prompt": " ".join(words[:n_words]), "n_predict": n_predict, "temperature": 0,
                "cache_prompt": True, "ignore_eos": True}
        req = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                t = json.loads(r.read().decode("utf-8", errors="replace")).get("timings", {})
        except Exception as e:
            print("  FAIL at %d words: %s" % (n_words, e))
            return rows
        ctx = (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)
        ms = t.get("predicted_per_token_ms") or 0
        dn, da = int(t.get("draft_n") or 0), int(t.get("draft_n_accepted") or 0)
        rows.append({"ctx": ctx, "ms": ms, "draft_n": dn, "draft_accepted": da,
                     "predicted_n": int(t.get("predicted_n") or 0),
                     "prompt_per_second": t.get("prompt_per_second") or 0})
        spec = ""
        if dn:
            spec = "  draft %d/%d accepted = %.0f%%" % (da, dn, 100.0 * da / dn)
        print("  context %6d tokens (prefill %5d at %6.1f t/s): %7.2f ms/token = %5.2f tok/s%s" % (
            ctx, t.get("prompt_n") or 0, t.get("prompt_per_second") or 0, ms, 1000.0 / ms if ms else 0, spec))
    if len(rows) >= 2 and rows[-1]["ctx"] > rows[0]["ctx"]:
        a, b = rows[0], rows[-1]
        print("  slope: %.3f ms per token per 1000 context tokens (%d -> %d)" % (
            (b["ms"] - a["ms"]) / (b["ctx"] - a["ctx"]) * 1000, a["ctx"], b["ctx"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[], help="config name from CONFIGS (repeatable)")
    ap.add_argument("--name", help="label for the log file (default: the config name)")
    ap.add_argument("--set", nargs="*", default=[], help="profile overrides, k=v")
    ap.add_argument("--env", nargs="*", default=[], help="extra environment overrides, K=V")
    # one string, shlex-split: argparse cannot take values that start with '-' as nargs items
    ap.add_argument("--argv-add", default="",
                    help='extra llama-server flags, appended after manager.argv, e.g. --argv-add="-ctkd q8_0"')
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n", type=int, default=64, help="tokens to decode at each depth")
    ap.add_argument("--words", default="700,28800", help="prefix lengths in words (~3.4 tokens each)")
    ap.add_argument("--runtime", help="launch from this runtime directory instead of the production one")
    ap.add_argument("--keep", action="store_true", help="probe the running server, launch nothing")
    ap.add_argument("--model", help="model id to launch (default: the first in the catalog)")
    ap.add_argument("--gen", type=int, default=0,
                    help="also write N deterministic tokens to logs/lab-<name>-gen.txt, to diff against another build")
    ap.add_argument("--gen-prefix", help="file whose first --gen-prefix-chars characters precede the gen prompt, "
                                         "so speculation is measured at depth against real prose rather than the "
                                         "random words the depth probe uses (those are undraftable and make "
                                         "acceptance meaningless)")
    ap.add_argument("--gen-prefix-chars", type=int, default=340000)
    ap.add_argument("--restore", action="store_true", help="restart the managed production server at the end")
    args = ap.parse_args()

    rnd = random.Random(20260918)
    steps = [int(w) for w in args.words.split(",")]
    words = ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9)))
             for _ in range(max(steps))]

    (ROOT / "logs").mkdir(exist_ok=True)      # a fresh checkout has no logs/ yet
    model_id = args.model or (default_model_id() if not args.keep or args.restore else None)

    results = {}
    if args.keep:
        print("== running server (no launch)")
        results["running"] = probe(args.port, steps, args.n, words)
    else:
        import manager
        for cfg_name in args.config or ["base"]:
            if cfg_name not in CONFIGS:
                print("unknown config %r; known: %s" % (cfg_name, ", ".join(sorted(CONFIGS))))
                return 2
            label = args.name or cfg_name
            print("== %s" % label)
            # only the production port is the app's server to stop; on a test port nothing of the user's is touched
            if args.port == 8080:
                call("stop", {})
            if not wait_vram_free():
                print("  VRAM never freed; something else is holding the carve")
                return 1
            pr = call("profile", {"id": model_id})["data"]["profile"]
            pr.update(PROFILE_OVERRIDES)
            for kv in args.set:
                k, v = kv.split("=", 1)
                pr[k] = parse(v)
            m = manager.model_by_id(model_id)
            # the default is the catalog's first model, which is whatever sorts first on this machine
            print("  model: %s" % m["path"])
            cfg = manager.validate_profile(pr, m)
            cmd = manager.argv(m, cfg)
            if args.runtime:
                cmd[0] = str((ROOT / args.runtime / "llama-server.exe").resolve())
            # the server listens where the probe looks: manager.argv names the production port
            if "--port" in cmd:
                cmd[cmd.index("--port") + 1] = str(args.port)
            cmd += shlex.split(args.argv_add)
            env = manager.runtime_environment(cfg)
            env.update(CONFIGS[cfg_name])
            for kv in args.env:
                k, v = kv.split("=", 1)
                env[k] = v
            log = ROOT / "logs" / ("lab-%s-server.log" % label)
            proc = subprocess.Popen(cmd, stdout=log.open("wb"), stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, cwd=str(ROOT))
            try:
                if not wait_ready(args.port):
                    print("  server never became ready, see", log)
                    continue
                results[label] = probe(args.port, steps, args.n, words)
                if args.gen:
                    question = ("Explain how a B-tree keeps its depth balanced, then give a short "
                                "worked example of inserting into a full node.")
                    prompt = question
                    if args.gen_prefix:
                        doc = Path(args.gen_prefix).read_text(encoding="utf-8", errors="replace")
                        prompt = doc[:args.gen_prefix_chars] + "\n\n" + question
                    # ignore_eos: a raw document prefix ends the sequence as soon as the model gets the
                    # chance, and one token tells us nothing about acceptance at depth
                    body = {"prompt": prompt, "n_predict": args.gen, "temperature": 0, "seed": 1,
                            "cache_prompt": False, "ignore_eos": True}
                    req = urllib.request.Request("http://127.0.0.1:%d/completion" % args.port,
                                                 data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        resp = json.loads(r.read().decode("utf-8", errors="replace"))
                    text = resp.get("content", "")
                    (ROOT / "logs" / ("lab-%s-gen.txt" % label)).write_text(text, encoding="utf-8")
                    t = resp.get("timings", {})
                    dn, da = int(t.get("draft_n") or 0), int(t.get("draft_n_accepted") or 0)
                    npred = int(t.get("predicted_n") or 0)
                    # real text, unlike the random-word depth probe, so the draft counters mean something
                    print("  gen: %d tokens at %.2f ms/token = %.2f tok/s%s; wrote logs/lab-%s-gen.txt" % (
                        npred, t.get("predicted_per_token_ms") or 0, t.get("predicted_per_second") or 0,
                        ("; draft %d/%d accepted = %.0f%%, %.2f tokens per pass" % (
                            da, dn, 100.0 * da / dn, npred / max(1, npred - da))) if dn else "", label))
                    if label in results and results[label]:
                        results[label][-1]["gen"] = {"predicted_n": npred, "ms": t.get("predicted_per_token_ms"),
                                                     "draft_n": dn, "draft_accepted": da}
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
            print("  log: %s" % log.relative_to(ROOT))

    if args.restore:
        print("== restoring managed production server")
        wait_vram_free()
        print("  start:", call("start", {"id": model_id}).get("ok"))
        for _ in range(90):
            if call("status", {})["data"].get("status") == "ready":
                print("  production: ready")
                break
            time.sleep(10)
    (ROOT / "logs" / "lab-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
