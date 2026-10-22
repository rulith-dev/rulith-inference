#include "pack.inc"
#include "prefix.h"
#include <vector>
#include <thread>
#include <cstdlib>
#include "models.h"

#include <cmath>

// strixllama: rank of the draft head's low-rank pre-score (nextn.lr_proj / lr_scores)
static constexpr int64_t QWEN4EXP_MTP_LR_RANK = 512;
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>
#include "block-graph.inc"

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

static const llama_model & qwen4exp_shared_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("QWEN4EXP MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model (-md), not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("QWEN4EXP MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    {   // The converted GGUF leaves the nextn/MTP layer's compress ratio at 0, but the MTP sidecar ships
        // blk.N.indexer.* and Halogen runs that layer with the same sparse attention as the trunk (one
        // k_attn_qs_bt4x call per target chunk for it). With LLAMA_MTP_QSA=1 inherit the trunk's ratio.
        static const bool mtp_qsa = getenv("LLAMA_MTP_QSA") && atoi(getenv("LLAMA_MTP_QSA")) != 0;
        if (mtp_qsa && hparams.n_layer_nextn > 0 && hparams.indexer_head_size > 0) {
            int32_t trunk_r = 0;
            for (uint32_t j = 0; j < hparams.n_layer(); ++j) {
                if (hparams.dsv4_compress_ratios[j] > 0) { trunk_r = hparams.dsv4_compress_ratios[j]; }
            }
            for (uint32_t j = hparams.n_layer(); j < hparams.n_layer_all && trunk_r > 0; ++j) {
                if (hparams.dsv4_compress_ratios[j] == 0) {
                    hparams.dsv4_compress_ratios[j] = trunk_r;
                    LLAMA_LOG_INFO("%s: LLAMA_MTP_QSA: layer %u compress ratio 0 -> %d\n", __func__, j, trunk_r);
                }
            }
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);

    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
        // strixllama: an EXL3 row codec table (ExLlamaV3's n-gram codec) comes with its per-head bias
        if (per_layer_tok_embd && ggml_is_exl3_row(per_layer_tok_embd->type)) {
            per_layer_tok_embd_b = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "bias"),
                                                 { hparams.ple_head_dim, hparams.ple_n_heads }, 0);
        }

        ple_reader = load_lazy_reader(ml, ple_name.c_str(), per_layer_tok_embd);
    }

    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { hc_dim }, flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        // strixllama: optional low-rank pre-score of the LM head for the draft (tools/make_draft_head.py --lowrank)
        layer.nextn.lr_proj   = create_tensor(tn(LLM_TENSOR_NEXTN_LR_PROJ,   "weight", il), { n_embd, QWEN4EXP_MTP_LR_RANK }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.lr_scores = create_tensor(tn(LLM_TENSOR_NEXTN_LR_SCORES, "weight", il), { QWEN4EXP_MTP_LR_RANK, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    if (hc_head_norm == nullptr) {
        throw std::runtime_error("this model is an MTP draft head without a trunk; "
                                 "load it as a draft of its target model (-md), not on its own");
    }
    return std::make_unique<graph>(*this, params);
}

// DFlash drafters read the layer inputs as [n_embd, T] rows, so collapse the streams
static ggml_tensor * hc_stream_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n_embd = x->ne[0];
    const int64_t hc     = x->ne[1];
    const int64_t nt     = x->ne[2];

    ggml_tensor * acc = ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], s*x->nb[1]));
    }

    return ggml_scale(ctx, acc, 1.0f / (float) hc);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    const char * pack_env = getenv("LLAMA_HC_PACK_DI");
    const bool pack_di = pack_env && atoi(pack_env) != 0 && nt >= 128 && inject &&
        loras->empty() && w_down->type == GGML_TYPE_IQ4_NL && w_down->type == w_inject->type &&
        ggml_is_matrix(w_down) && ggml_is_matrix(w_inject) &&
        ggml_is_contiguous(w_down) && ggml_is_contiguous(w_inject) &&
        w_down->ne[0] == w_inject->ne[0];
    ggml_tensor * lo;
    if (pack_di) {
        ggml_build_forward_expand(gf, xn);
        ggml_tensor * weights = ggml_concat(ctx0, w_down, w_inject, 1);
        cb(weights, "hc_down_inject_weights", il);
        ggml_tensor * projected = ggml_mul_mat(ctx0, weights, xn);
        cb(projected, "hc_down_inject", il);
        lo = ggml_cont(ctx0, ggml_view_2d(ctx0, projected, w_down->ne[1], nt, projected->nb[1], 0));
        *inject = ggml_cont(ctx0, ggml_view_2d(ctx0, projected, w_inject->ne[1], nt,
                projected->nb[1], w_down->ne[1] * sizeof(float)));
        ggml_build_forward_expand(gf, *inject);
    } else {
        lo = build_lora_mm(w_down, xn);
    }
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    mixed = ggml_cont(ctx0, mixed);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        if (!pack_di) {
            *inject = build_lora_mm(w_inject, xn);
        }
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    ggml_build_forward_expand(gf, residual);
    ggml_build_forward_expand(gf, block_out);
    ggml_build_forward_expand(gf, inject);

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);
    ggml_build_forward_expand(gf, w);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                mctx_idx->get_kv_off() == inp->mctx->get_attn()->get_kv_off() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = hc_stream_mean(ctx0, res_hc);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

static bool qwen4exp_pos_scalar(const llama_ubatch & ubatch);

// strixllama: IndexShare (see llama_memory_hybrid_idx::share_sel). A catch-up keeps the selections of its tokens (at most
// QWEN4EXP_SHARE_KEEP_ALL of them - a verify's catch-up, several slots at once; a prompt keeps none) when a sequence's
// latest kept selection is more than LLAMA_MTP_INDEX_SHARE_REFRESH positions behind. A draft step attends to that
// selection plus up to QWEN4EXP_SHARE_EXTRA cells of the positions since: the refresh distance, the catch-up, the draft
// steps. The gather pads the selection to 2304 cells anyway, so the extra columns are free up to 253.
static constexpr int64_t QWEN4EXP_SHARE_KEEP_ALL = 32;
static constexpr int64_t QWEN4EXP_SHARE_EXTRA    = llama_memory_hybrid_idx::share_extra;

// the refresh distance, at most the widest gap a draft step's extra columns cover
static llama_pos qwen4exp_share_refresh() {
    static const llama_pos v = getenv("LLAMA_MTP_INDEX_SHARE_REFRESH") ? atoi(getenv("LLAMA_MTP_INDEX_SHARE_REFRESH")) : 32;
    return std::max<llama_pos>(0, std::min<llama_pos>(v, llama_memory_hybrid_idx::share_max_gap));
}

// whether the MTP layer's graph for this ubatch takes part in IndexShare: the memory keeps selections, the layer is
// sparse, one stream, 1-D positions, and a view long enough to pay. A draft step's dense read of a short view costs
// next to nothing, so below LLAMA_MTP_INDEX_SHARE_MIN_KV cells nothing is kept and the draft runs as without it
// (measured 09-28: -0.6% a pass at 30K, -3.6% at 86K, -6.5% at 212K; the old every-catch-up design lost 4% at 4K).
static bool qwen4exp_mtp_share_on(const llama_memory_hybrid_idx_context * mctx_hyb, const llama_ubatch & ubatch,
        const llama_hparams & hparams) {
    static const int64_t min_kv = getenv("LLAMA_MTP_INDEX_SHARE_MIN_KV") ? atoll(getenv("LLAMA_MTP_INDEX_SHARE_MIN_KV")) : 32768;
    const auto * mem = mctx_hyb ? mctx_hyb->get_mem() : nullptr;
    const uint32_t il = hparams.n_layer();
    return mem && mem->share_sel() != nullptr && il < hparams.n_layer_all && hparams.dsv4_compress_ratios[il] > 0 &&
        mctx_hyb->get_idx() != nullptr && mctx_hyb->get_n_stream() == 1 && qwen4exp_pos_scalar(ubatch) &&
        (int64_t) mctx_hyb->get_idx()->get_n_kv() >= min_kv;
}

// whether an MTP ubatch without outputs runs the indexer to keep its tokens' selections
static bool qwen4exp_mtp_capture(const llama_memory_hybrid_idx_context * mctx_hyb, const llama_ubatch & ubatch,
        const llama_hparams & hparams) {
    if (!qwen4exp_mtp_share_on(mctx_hyb, ubatch, hparams) || (int64_t) ubatch.n_tokens > QWEN4EXP_SHARE_KEEP_ALL) {
        return false;
    }
    const int64_t r = hparams.dsv4_compress_ratios[hparams.n_layer()];
    if ((int64_t) mctx_hyb->get_idx()->get_n_kv() <= (int64_t) hparams.indexer_top_k + r - 1) {
        return false;
    }
    return mctx_hyb->get_mem()->share_capture_due(ubatch, qwen4exp_share_refresh());
}

// the choice qwen4exp_mtp_capture made when the graph was built, re-checked before the graph is reused
class llama_model_qwen4exp::llm_graph_input_mtp_capture : public llm_graph_input_i {
public:
    explicit llm_graph_input_mtp_capture(bool capture) : capture(capture) {}
    virtual ~llm_graph_input_mtp_capture() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        return qwen4exp_mtp_capture(m, params.ubatch, params.hparams) == capture;
    }

    const bool capture;
};

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    // a prompt's image rows come as embeddings (upstream's draft-mtp adds them with their rows since #30397)

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // the h rows come as the batch's state rows (llama_batch_ext's embd_state), the image rows as its embeddings
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    if (tok_embd_w == nullptr) {
        tok_embd_w = qwen4exp_shared_model(cparams, model, "token_embd.weight").tok_embd;
    }
    ggml_tensor * tok_embd   = ubatch.token ? ggml_get_rows(ctx0, tok_embd_w, inp->tokens) : inp->embd;
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // with LLAMA_MTP_QSA the MTP context is a hybrid-idx memory, so the draft head can use the same sparse
    // attention path as the trunk (this is what Halogen does: one sparse attention call per target chunk)
    static const bool mtp_qsa = getenv("LLAMA_MTP_QSA") && atoi(getenv("LLAMA_MTP_QSA")) != 0;
    llm_graph_input_attn_kv * inp_attn = nullptr;
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;
    if (mtp_qsa && hparams.indexer_head_size > 0) {
        auto * inp_hyb = build_inp_mem_hybrid();
        const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);
        if (m->get_idx() != nullptr) {
            mctx_hyb = m;
            inp_attn = inp_hyb->get_attn();
            GGML_ASSERT(m->get_idx()->get_n_kv() == m->get_attn()->get_n_kv() &&
                    m->get_idx()->get_kv_off() == m->get_attn()->get_kv_off() &&
                    "the indexer cache must track the attention cache cell for cell");
            // the MTP graph has no recurrent layers, so the hybrid input's recurrent tensors are never used and the
            // allocator skips them -- set_input would then hit a null buffer. Give them a trivial use.
            auto * rs = inp_hyb->get_recr();
            for (ggml_tensor * t : { rs->s_copy, rs->s_copy_main, rs->s_copy_extra }) {
                if (t) { ggml_build_forward_expand(gf, ggml_scale(ctx0, ggml_cast(ctx0, t, GGML_TYPE_F32), 0.0f)); }
            }
        }
    }
    if (!inp_attn) { inp_attn = build_attn_inp_kv(); }

    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // per stream, not pooled: pooling before the projection discards the hyper-connection residual.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // The eh projection sees concat as [2*n_embd, hc, n_tokens], so mul_mat batches over the
    // token axis with only hc(=4) columns per batch and dispatches a 4-column MMVQ. Flattening the
    // hc and token axes into a single column axis exposes one wide MMQ GEMM instead. The weight is
    // shared across the merged axes, so the arithmetic is identical up to the kernel's own rounding;
    // this is the MTP draft head, so it cannot change the target prefill logits.
    const char * eh_flat_env = getenv("LLAMA_MTP_EH_FLATTEN");
    const bool eh_flatten = !eh_flat_env || atoi(eh_flat_env) != 0;
    ggml_tensor * res_hc;
    if (eh_flatten) {
        ggml_tensor * concat_flat = ggml_reshape_2d(ctx0, concat, concat->ne[0], concat->ne[1] * concat->ne[2]);
        ggml_tensor * res_flat    = build_lora_mm(layer.nextn.eh_proj, concat_flat, layer.nextn.eh_proj_s);
        res_hc = ggml_reshape_3d(ctx0, res_flat, res_flat->ne[0], concat->ne[1], concat->ne[2]);
    } else {
        res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    }
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // strixllama: a ubatch without outputs - the catch-up after a verify, a prompt - only fills the draft's caches: its
    // attention output reaches no output row (the FFN and the head see only those), so K, V and the indexer keys are
    // stored and the query, the attention and wo are not built (LLAMA_MTP_STORE_ONLY=0 builds them)
    static const bool store_only_on = !getenv("LLAMA_MTP_STORE_ONLY") || atoi(getenv("LLAMA_MTP_STORE_ONLY")) != 0;
    const bool store_only = store_only_on && n_outputs == 0 && inp_out_ids != nullptr;

    ggml_tensor * Qcur = nullptr;
    ggml_tensor * gate = nullptr;
    if (!store_only) {
    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);
    }

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    if (Qcur) {
    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    }
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (mctx_hyb) {
        // the converted GGUF leaves compress_ratios[nextn] = 0, but the sidecar ships blk.N.indexer.* and the
        // reference runs this layer sparsely; fall back to the trunk ratio
        const int64_t r        = hparams.dsv4_compress_ratios[il];   // patched at load when LLAMA_MTP_QSA is set
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = (int64_t) hparams.indexer_top_k + r - 1;
        ggml_tensor * top_k = nullptr;
        // Sparse draft attention only pays when there are enough queries to amortise the indexer
        // (scorer over every block, top-k, block selection). At decode sizes (1-4 queries) the indexer
        // costs more than dense attention over the whole cache, and qsa3 declines those shapes anyway
        // (its own q->ne[1] < 128 gate), so the sparse path would land on the slow split-D kernel.
        // build_qsa_store_k still runs, so the indexer cache stays complete for later sparse ubatches.
        static const int64_t mtp_min_t = getenv("LLAMA_MTP_QSA_MIN_T") ? atoll(getenv("LLAMA_MTP_QSA_MIN_T")) : 128;
        if (qwen4exp_mtp_share_on(mctx_hyb, ubatch, hparams) && cparams.mtp_share_reuse && !store_only) {
            // strixllama: IndexShare - a draft step attends to the selection a catch-up kept, plus the positions since
            build_qsa_store_k(mctx_hyb, cur, il);
            top_k = build_mtp_share_reuse(mctx_hyb);
        } else if (store_only) {
            // strixllama: IndexShare - a catch-up keeps its tokens' selections when the latest kept one is too far behind
            // (qwen4exp_mtp_capture); the graph fixes the choice, so an input re-checks it before the graph is reused
            const bool capture = qwen4exp_mtp_capture(mctx_hyb, ubatch, hparams);
            if (capture) {
                ggml_tensor * sel = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, il);
                GGML_ASSERT(sel->ne[0] == mctx_hyb->get_mem()->share_width() && sel->ne[1] == n_tokens &&
                        sel->ne[2] == 1 && sel->ne[3] == 1);
                build_mtp_share_capture(mctx_hyb, sel);
            } else if (r > 0) {
                build_qsa_store_k(mctx_hyb, cur, il);
            }
            res->add_input(std::make_unique<llm_graph_input_mtp_capture>(capture));
        } else if (r > 0 && n_kv_idx > width && (int64_t) n_tokens >= mtp_min_t) {
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, il);
        } else if (r > 0) {
            build_qsa_store_k(mctx_hyb, cur, il);
        }
        if (store_only) {
            build_mtp_store_kv(inp_attn, Kcur, Vcur, il);
        } else if (top_k) {
            cur = build_attn_qsa(inp_attn, Qcur, Kcur, Vcur, top_k, kq_scale, il);
        } else {
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
    } else if (store_only) {
        build_mtp_store_kv(inp_attn, Kcur, Vcur, il);
    } else {
    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }

    if (store_only) {
        // no row is an output: every node from here on is empty, so the rows are taken before the attention's place
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    } else {
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);
    }

    if (inp_out_ids) {
        if (!store_only) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        }
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // no res->t_embd: it is n_embd wide, but the context sizes that buffer by n_embd_out.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    if (head_w == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output.weight");
        head_w = other.output;
        head_s = other.output_s;
        GGML_ASSERT(head_w && "QWEN4EXP MTP: the target model has no LM head to borrow");
    }

    // strixllama: the draft needs only the head's top tokens. With nextn.lr_proj / lr_scores in the draft file a
    // rank-512 pre-score over the vocabulary (Q8_0, 135 MB against the 338 MB IQ4_XS head) picks STRIX_MTP_LR_K
    // candidates, and only their rows of the head are multiplied exactly; every other token gets -inf. The draft
    // samples from the candidates and the target verifies as before, so outputs stay the target's
    static const int lr_k = getenv("STRIX_MTP_LR_K") ? atoi(getenv("STRIX_MTP_LR_K")) : 512;
    const int64_t nt = cur->ne[1];
    // (an EXL3 head has no rows to gather - its output rotation mixes 128 rows - so it is multiplied in full)
    if (layer.nextn.lr_proj && layer.nextn.lr_scores && lr_k > 0 && head_s == nullptr && nt > 0 && !ggml_is_exl3(head_w->type)) {
        ggml_tensor * q   = ggml_mul_mat(ctx0, layer.nextn.lr_proj, cur);                        // [rank, nt]
        ggml_tensor * sc  = ggml_mul_mat(ctx0, layer.nextn.lr_scores, q);                        // [n_vocab, nt]
        cb(sc, "mtp_lr_scores", -1);
        ggml_tensor * idx = ggml_top_k(ctx0, sc, lr_k);                                          // [K, nt]
        ggml_tensor * rows = ggml_get_rows(ctx0, head_w, ggml_reshape_1d(ctx0, idx, lr_k * nt)); // [n_embd, K*nt]
        rows = ggml_reshape_3d(ctx0, rows, n_embd, lr_k, nt);
        ggml_tensor * ex  = ggml_mul_mat(ctx0, rows, ggml_reshape_3d(ctx0, cur, n_embd, 1, nt)); // [K, 1, nt]
        ggml_tensor * full = ggml_fill(ctx0, sc, -INFINITY);
        cur = ggml_set_rows(ctx0, ggml_reshape_3d(ctx0, full, 1, sc->ne[0], nt), ggml_reshape_3d(ctx0, ex, 1, lr_k, nt), idx);
        cur = ggml_reshape_2d(ctx0, cur, sc->ne[0], nt);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    // strixllama: a ragged ubatch keeps one column per token; build_layer_attn_linear takes each group's columns
    const int64_t n_seqs       = ubatch.ragged() ? 1 : ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.ragged() ? n_tokens : ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
// A batch that carries BOTH tokens and embeddings is the MTP draft head: common_speculative's draft-mtp
// impl fills batch.token and batch.embd together (the head consumes the target's h_nextn row per token).
// The original `!ubatch.embd` guard was aimed at pure-embedding (vision) batches, which are already
// excluded by requiring ubatch.token; keeping it shut the draft out of block selection, and with it out of
// the maskless/packed-key layout and the qsa3 attention kernel. Set LLAMA_QSA_TOKEN_EMBD=0 to restore it.
static bool qwen4exp_qsa_embd_ok(const llama_ubatch & ubatch) {
    if (!ubatch.embd) { return true; }
    const char * v = getenv("LLAMA_QSA_TOKEN_EMBD");
    return !v || atoi(v) != 0;
}

// strixllama: the most queries the decode gather takes (qwen4exp_gather_attn); 32 covers eight slots verifying a draft
// of 3 each, and above that the gather reads more than it saves
static int64_t qwen4exp_gather_max_t() {
    static const int64_t v = getenv("LLAMA_QSA_DECODE_GATHER_MAX_T") ? (int64_t) atoll(getenv("LLAMA_QSA_DECODE_GATHER_MAX_T")) : 32;
    return v;
}

// strixllama: a ubatch between the two kernels that read the selected indices - more queries than the decode gather
// takes, fewer than the 128 the qsa3 prefill kernel needs. The selection went to the generic flash attention with the
// plain causal mask, which ignores it: dense attention over every visible cell instead of the model's sparse one.
// Such a ubatch takes the plain top-k and the top-k mask instead (the block selection's -1 sentinels are for the
// kernels only; the mask's set_rows cannot take them)
static bool qwen4exp_qsa_between_kernels(int64_t n_tps) {
    return n_tps > qwen4exp_gather_max_t() && n_tps < 128;
}

static bool qwen4exp_use_block_selection(bool blk_bias, int64_t n_stream, int64_t ratio, int64_t n_kv,
        const llama_ubatch & ubatch, const llama_cparams & cparams, const llama_hparams & hparams) {
    const char * block_env=getenv("LLAMA_QSA_BLOCK_SELECTION");
    const char * direct_env=getenv("LLAMA_QSA_DIRECT_INDICES");
    return block_env && atoi(block_env)!=0 && direct_env && atoi(direct_env)!=0 &&
        n_stream > 0 && !qwen4exp_qsa_between_kernels(ubatch.n_tokens / n_stream) &&
        blk_bias && n_stream==1 && ratio>1 && hparams.indexer_top_k%ratio==0 &&
        n_kv>hparams.indexer_top_k+ratio-1 && n_kv<=16777216 && ubatch.token && qwen4exp_qsa_embd_ok(ubatch) &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias==0.0f &&
        !hparams.attn_soft_cap && hparams.n_embd_head_k()==256 && hparams.n_embd_head_v()==256;
}

// the mrope sections make every ubatch "2-D" (n_pos > 1); text has all rows equal, images do not
static bool qwen4exp_pos_scalar(const llama_ubatch & ubatch) {
    if (!ubatch.pos || ubatch.n_tokens == 0) { return false; }
    // a text token comes as [p, p, p, 0] since llama_batch_ext (the fourth section is unused here): three decide
    for (uint32_t axis = 1; axis < std::min<uint32_t>(ubatch.n_pos, 3); ++axis) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) { return false; }
        }
    }
    return true;
}

static bool qwen4exp_qsa_flag(const char *name) {
    const char *value=getenv(name);
    return value && atoi(value)!=0;
}

// strixllama: may an attention over n_kv cells go without the dense mask? Only the qsa3 kernel honours the selected
// indices without one, so this must mirror what it takes (ggml_cuda_flash_attn_ext_qsa_supported): past that the
// generic flash-attention kernels run and ignore the indices, and without the mask every query attends to every cell
// of the window - other conversations and later positions included. Up to 0.2.7 the kernel stopped at 262140 cells
// (16-bit block numbers), and 0.2.7's ragged ubatches, a prompt chunk next to three long answers, spanned more once
// the four held ~262K cells together: ~20 ms a token, for a wrong result.
static bool qwen4exp_qsa_maskless_ok(int64_t n_kv, int64_t n_tps) {
    return n_tps >= 128 && n_kv <= INT32_MAX - 3 && qwen4exp_qsa_flag("LLAMA_QSA_NO_DENSE_MASK");
}

static int64_t qwen4exp_query_strip(int64_t n_tokens, int64_t n_stream);

// [QSA_SCORE_BOUNDS] Visible-prefix scoring (ported from 2026-09-09-sparse-prefill/qsa-score-bounds).
// For one sequence with unique non-negative positions the indexer enumerates complete groups in increasing
// logical block order, so a compressed column's ordinal cannot exceed its logical block number: every block
// visible to any query in a strip lies in the first (max_query_position + 1)/ratio columns. Holes and physical
// cell permutations do not invalidate that upper bound. Blocks past it are -inf in the visibility metadata
// anyway, so trimming them must not change the selection.
static std::vector<int64_t> qwen4exp_score_key_limits(const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch, int64_t blocks, int64_t strip, int64_t ratio, int64_t budget, bool compact,
        bool active_only = false) {
    // llama_context::graph_reserve builds a worst-case graph from a synthetic ubatch whose positions are all 0.
    // Bounding that graph would reserve compute buffers for a 4%-wide scorer and then execute full-width ones, so a
    // many-token ubatch whose positions are all identical is treated as synthetic and left unbounded.
    bool degenerate_pos = ubatch.pos && ubatch.n_tokens > 1;
    for (uint32_t i = 1; degenerate_pos && i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != ubatch.pos[0]) { degenerate_pos = false; }
    }
    if (!compact || ubatch.n_tokens<128 || degenerate_pos || !qwen4exp_qsa_flag("LLAMA_QSA_SCORE_BOUNDS") ||
            !mctx->qsa_position_prefix(ubatch, active_only)) {
        static unsigned off = 0;
        if (off++ < 2) { fprintf(stderr,"QSA_SCORE_BOUNDS inactive (compact=%d tokens=%u flag=%d prefix=%d)\n",
                (int) compact, ubatch.n_tokens, (int) qwen4exp_qsa_flag("LLAMA_QSA_SCORE_BOUNDS"),
                (int) mctx->qsa_position_prefix(ubatch, active_only)); }
        return {};
    }
    auto limits = qsa_prefix_limits(ubatch.pos,ubatch.n_tokens,strip,ratio,blocks,budget);
    {
        static unsigned hits = 0;
        if (hits++ < 10 && !limits.empty()) {
            int64_t sum = 0; for (auto v : limits) { sum += v; }
            llama_pos pmin = ubatch.pos[0], pmax = ubatch.pos[0];
            for (uint32_t i = 1; i < ubatch.n_tokens; ++i) { pmin = std::min(pmin, ubatch.pos[i]); pmax = std::max(pmax, ubatch.pos[i]); }
            fprintf(stderr,"QSA_SCORE_BOUNDS active tokens=%u pos=[%d..%d] blocks=%lld strips=%zu first=%lld last=%lld mean=%lld (%.0f%% of full)\n",
                    ubatch.n_tokens,(int)pmin,(int)pmax,(long long)blocks,limits.size(),(long long)limits.front(),(long long)limits.back(),
                    (long long)(sum/(int64_t)limits.size()), 100.0*double(sum)/double(blocks*(int64_t)limits.size()));
        }
    }
    return limits;
}

// strixllama: the rows of a block-key dirty list: the blocks a ubatch of n_tps tokens (per stream) completes, up to two per
// sequence at its ends, and per sequence the blocks of a pending span (llama_memory_hybrid_idx::kb_pending_max positions,
// in up to two runs beside the ubatch's own)
static int64_t qwen4exp_kb_dirty_max(int64_t n_tps, int64_t ratio, int64_t n_seqs) {
    return n_tps/ratio + 2*n_seqs + 2 + n_seqs*(llama_memory_hybrid_idx::kb_pending_max/ratio + 4);
}

class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        llama_memory_hybrid_idx::qsa_kb_inputs kb;
        kb.dirty_cells = kb_dirty_cells; kb.dirty_pos = kb_dirty_pos; kb.dirty_dst = kb_dirty_dst; kb.bid_rows = kb_bid_rows;
        const auto * kbp = kb_bid_rows ? &kb : nullptr;
        if (tail_idxs) {
            llama_memory_hybrid_idx::qsa_mixed_inputs mx;
            mx.seq_blk = seq_blk; mx.seq_tok = seq_tok;
            mctx->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kbp, seq_blk ? &mx : nullptr, active_only);
        } else {
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, kbp);
        }
        // a graph that rewrites every block key refreshes the cache when it runs, not when it is built:
        // llama_context::graph_reserve builds graphs that never execute
        if (kbp && kb_full) {
            mctx->kb_mark_full(*ubatch);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const bool blocks=qwen4exp_use_block_selection(blk_bias,n_stream,ratio,n_kv,
                params.ubatch,params.cparams,params.hparams);
        const bool scalar=blocks && params.hparams.n_swa==0 && mctx->qsa_scalar_visibility(params.ubatch);
        // strixllama: a compact graph's block list may cover only the ubatch's sequences (qsa_active_blocks)
        const int64_t n_active = scalar && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA") ?
                mctx->qsa_active_blocks(params.ubatch, ratio, params.hparams.indexer_top_k/ratio) : 0;
        const int64_t n_blocks = n_active > 0 ? n_active : (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= active_only == (n_active > 0);

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0] == (compact ? n_blocks+params.ubatch.n_tokens : (blk_bias ? n_blocks : n_kv));
        res &= compact || bias->ne[1] == params.ubatch.n_tokens/n_stream;
        res &= (tail_idxs != nullptr) == blocks;
        res &= compact == (scalar && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA"));
        res &= maskless == (scalar && qwen4exp_qsa_maskless_ok(n_kv, params.ubatch.n_tokens/n_stream));
        // strixllama: the membership inputs exist for a compact ubatch of several sequences, sized by their count
        res &= (seq_blk != nullptr) == (compact && params.ubatch.n_seqs_unq > 1);
        if (seq_blk) {
            res &= seq_blk->ne[0] == params.ubatch.n_seqs_unq && seq_blk->ne[1] == n_blocks;
            res &= seq_tok->ne[0] == params.ubatch.n_seqs_unq && seq_tok->ne[1] == params.ubatch.n_tokens/n_stream;
        }
        if (tail_idxs) { res &= tail_idxs->ne[1] == params.ubatch.n_tokens/n_stream; }
        // [QSA_SCORE_BOUNDS] the trimmed widths are baked into the graph, so a reused graph must agree on them
        const int64_t next_strip=qwen4exp_query_strip(params.ubatch.n_tokens/n_stream,n_stream);
        const auto next_limits=qwen4exp_score_key_limits(mctx,params.ubatch,n_blocks,next_strip,ratio,
                params.hparams.indexer_top_k/ratio,scalar && compact,active_only);
        res &= score_strip==next_strip;
        res &= score_key_limits==next_limits;
        // strixllama: block-key cache shapes and mode are baked into the graph
        // the block-key cache is wired in only while no image cells are present, so a graph built
        // before one arrived cannot be reused after (it would take the ranked path and abort)
        res &= kb_dup == mctx->kb_pos_dup();
        if (kb_bid_rows) {
            res &= kb_full == mctx->kb_needs_full(params.ubatch);
            res &= kb_bid_rows->ne[0] == n_blocks;
            if (kb_dirty_dst) { res &= kb_dirty_dst->ne[0] == qwen4exp_kb_dirty_max(params.ubatch.n_tokens/n_stream, ratio, params.ubatch.n_seqs_unq); }
        }

        return res;
    }

    // strixllama: block-key cache inputs (null when the cache is off); kb_full = this graph rebuilds every key
    ggml_tensor * kb_dirty_cells = nullptr;   // I32 [ratio*dirty_max, 1]
    ggml_tensor * kb_dirty_pos   = nullptr;   // I32 [4*dirty_max]
    ggml_tensor * kb_dirty_dst   = nullptr;   // I32 [dirty_max]
    ggml_tensor * kb_bid_rows    = nullptr;   // I32 [n_blocks]
    bool          kb_full        = false;
    bool          kb_dup         = false;     // the memory held image cells when this graph was built

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    ggml_tensor * tail_idxs = nullptr;
    // strixllama: several sequences in a compact ubatch (llama_memory_hybrid_idx::qsa_mixed_inputs)
    ggml_tensor * seq_blk = nullptr;   // F32 [n_seqs_unq, n_blocks]
    ggml_tensor * seq_tok = nullptr;   // F32 [n_seqs_unq, n_tokens/n_stream]
    bool compact = false;
    bool maskless = false;
    bool active_only = false;   // the block list covers only the ubatch's own sequences
    int64_t score_strip = 0;
    std::vector<int64_t> score_key_limits;       // [QSA_SCORE_BOUNDS] per strip, empty = no bound

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
};

class llama_model_qwen4exp::llm_graph_input_qsa_k : public llm_graph_input_i {
public:
    llm_graph_input_qsa_k(const llama_memory_hybrid_idx_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_qsa_k() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        // strixllama: these keys are written without the indexer, so their blocks' cached keys are not refreshed
        mctx->kb_note_unrefreshed(*ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        return k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    ggml_tensor * k_idxs = nullptr;   // I32 [n_tokens]

    const llama_memory_hybrid_idx_context * mctx;
};

class llama_model_qwen4exp::llm_graph_input_mtp_share : public llm_graph_input_i {
public:
    llm_graph_input_mtp_share(const llama_memory_hybrid_idx_context * mctx, bool reuse) : mctx(mctx), reuse(reuse) {}
    virtual ~llm_graph_input_mtp_share() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const auto * mem  = mctx->get_mem();
        const auto * attn = mctx->get_attn();
        const int64_t kv_off = attn->get_kv_off();

        llama_host_write(off);
        ((float *) off->data)[0] = (float) kv_off;
        ((float *) off->data)[1] = (float) attn->get_n_kv();

        llama_host_write(rows);
        int32_t * r = (int32_t *) rows->data;
        const uint32_t n_tokens = ubatch->n_tokens;

        if (!reuse) {
            llama_host_write(dst);
            int32_t * d = (int32_t *) dst->data;
            // each sequence's last position in the ubatch
            std::map<llama_seq_id, llama_pos> last;
            std::map<llama_seq_id, uint32_t>  last_i;
            for (uint32_t i = 0; i < n_tokens; ++i) {
                const llama_seq_id s = ubatch->seq_id[i][0];
                if (!last.count(s) || ubatch->pos[i] >= last[s]) {
                    last[s]   = ubatch->pos[i];
                    last_i[s] = i;
                }
            }
            const llama_pos ring = (llama_pos) mem->share_ring_len();
            if ((int64_t) n_tokens == rows->ne[0]) {
                for (uint32_t i = 0; i < n_tokens; ++i) {
                    const llama_seq_id s = ubatch->seq_id[i][0];
                    const llama_pos    p = ubatch->pos[i];
                    r[i] = (int32_t) i;
                    // only the last `ring` positions of a sequence: two tokens must never write one row
                    if (p > last[s] - ring && s >= 0 && s < mem->share_n_seqs()) {
                        d[i] = mem->share_row(s, p);
                        mem->share_note(s, p);
                    } else {
                        d[i] = mem->share_scratch_row();
                    }
                }
            } else {
                int64_t k = 0;
                for (const auto & [s, i] : last_i) {
                    if (k >= rows->ne[0]) {
                        break;
                    }
                    r[k] = (int32_t) i;
                    if (s >= 0 && s < mem->share_n_seqs()) {
                        d[k] = mem->share_row(s, ubatch->pos[i]);
                        mem->share_note(s, ubatch->pos[i]);
                    } else {
                        d[k] = mem->share_scratch_row();
                    }
                    ++k;
                }
                for (; k < rows->ne[0]; ++k) {
                    r[k] = 0;
                    d[k] = mem->share_scratch_row();
                }
            }
            return;
        }

        llama_host_write(extra);
        int32_t * e = (int32_t *) extra->data;
        const int64_t n_extra = extra->ne[0];
        const int64_t n_kv    = attn->get_n_kv();
        std::vector<int64_t> since;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];
            // the latest selection kept before this position, and the cells of the positions after it up to this one:
            // the tokens accepted since, the ones drafted in this loop, this one (its cell is already in the cache)
            const llama_pos cap = mem->share_latest(s, p);
            r[i] = cap >= 0 ? mem->share_row(s, cap) : mem->share_none_row();
            mem->share_cells(s, cap >= 0 ? cap + 1 : std::max<llama_pos>(0, p - (llama_pos) n_extra + 1), p, since);
            // the most recent positions if there are more than the columns hold
            const size_t first = since.size() > (size_t) n_extra ? since.size() - (size_t) n_extra : 0;
            for (int64_t k = 0; k < n_extra; ++k) {
                const size_t  j = first + (size_t) k;
                const int64_t c = j < since.size() ? since[j] - kv_off : -1;
                e[i*n_extra + k] = c >= 0 && c < n_kv ? (int32_t) c : -1;
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        if (params.cparams.mtp_share_reuse != reuse) {
            return false;
        }
        const int64_t n_tokens = params.ubatch.n_tokens;
        if (reuse) {
            return rows->ne[0] == n_tokens && extra->ne[1] == n_tokens;
        }
        return rows->ne[0] == (n_tokens <= QWEN4EXP_SHARE_KEEP_ALL ? n_tokens : (int64_t) params.ubatch.n_seqs_unq);
    }

    ggml_tensor * off   = nullptr;   // F32 [2]: the first cell of the graph's view of the cache, and the view's length
    ggml_tensor * rows  = nullptr;   // I32 capture: the ubatch rows kept [n_keep]; reuse: each token's share row [n_tokens]
    ggml_tensor * dst   = nullptr;   // I32 [n_keep]: capture: the share rows written (scratch for rows not kept)
    ggml_tensor * extra = nullptr;   // I32 [QWEN4EXP_SHARE_EXTRA, n_tokens]: reuse: view cells of the positions since, -1 padding

    const llama_memory_hybrid_idx_context * mctx;
    const bool reuse;
};

void llama_model_qwen4exp::graph::build_mtp_store_kv(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        int                       il) {
    // rotated before a quantized cache and stored exactly as build_attn and build_attn_qsa do it
    if (inp->self_k_rot) {
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));
}

void llama_model_qwen4exp::graph::build_mtp_share_capture(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           top_k) {
    const auto * mem = mctx_hyb->get_mem();
    const int64_t w = mem->share_width();
    GGML_ASSERT(top_k->ne[0] == w && top_k->ne[1] == n_tokens && top_k->ne[2] == 1 && top_k->ne[3] == 1);

    const int64_t n_keep = n_tokens <= QWEN4EXP_SHARE_KEEP_ALL ? (int64_t) n_tokens : (int64_t) ubatch.n_seqs_unq;
    auto inp = std::make_unique<llm_graph_input_mtp_share>(mctx_hyb, false);
    inp->off  = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2);
    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_keep);
    inp->dst  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_keep);
    ggml_set_input(inp->off);
    ggml_set_input(inp->rows);
    ggml_set_input(inp->dst);

    // a view cell c becomes absolute c + 1 (the view moves between graphs), "no cell" (-1) becomes 0
    ggml_tensor * off   = ggml_view_1d(ctx0, inp->off, 1, 0);
    ggml_tensor * sel   = ggml_cast(ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, top_k), w, n_tokens), GGML_TYPE_F32);
    ggml_tensor * valid = ggml_step(ctx0, ggml_scale_bias(ctx0, sel, 1.0f, 0.5f));
    ggml_tensor * abs1  = ggml_mul(ctx0, ggml_add(ctx0, ggml_scale_bias(ctx0, sel, 1.0f, 1.0f), off), valid);
    ggml_tensor * keep  = ggml_get_rows(ctx0, abs1, inp->rows);
    cb(keep, "mtp_share_keep", -1);

    ggml_tensor * t    = mem->share_sel();
    ggml_tensor * view = ggml_view_2d(ctx0, t, t->ne[0], t->ne[1], t->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, view, keep, inp->dst));

    res->add_input(std::move(inp));
}

ggml_tensor * llama_model_qwen4exp::graph::build_mtp_share_reuse(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mem = mctx_hyb->get_mem();
    const int64_t w = mem->share_width();

    auto inp = std::make_unique<llm_graph_input_mtp_share>(mctx_hyb, true);
    inp->off   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2);
    inp->rows  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->extra = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, QWEN4EXP_SHARE_EXTRA, n_tokens);
    ggml_set_input(inp->off);
    ggml_set_input(inp->rows);
    ggml_set_input(inp->extra);

    ggml_tensor * off  = ggml_view_1d(ctx0, inp->off, 1, 0);
    ggml_tensor * n_kv = ggml_view_1d(ctx0, inp->off, 1, ggml_element_size(inp->off));
    ggml_tensor * t    = mem->share_sel();
    ggml_tensor * view = ggml_view_2d(ctx0, t, t->ne[0], t->ne[1], t->nb[1], 0);
    ggml_tensor * g    = ggml_get_rows(ctx0, view, inp->rows);                    // [w, n_tokens]: absolute + 1, 0 = none
    // back to cells of this graph's view: d = absolute - off + 1 is a cell of the view when 1 <= d <= n_kv. "None" and a
    // cell outside the view (a row the bookkeeping failed to drop) both become -1: the gather must never read past the view
    ggml_tensor * d     = ggml_sub(ctx0, g, off);
    ggml_tensor * valid = ggml_mul(ctx0,
            ggml_step(ctx0, ggml_scale_bias(ctx0, d, 1.0f, -0.5f)),
            ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, d, n_kv), -1.0f, 0.5f)));
    ggml_tensor * rel  = ggml_scale_bias(ctx0, ggml_mul(ctx0, d, valid), 1.0f, -1.0f);
    rel = ggml_cast(ctx0, rel, GGML_TYPE_I32);
    ggml_tensor * sel  = ggml_concat(ctx0, rel, inp->extra, 0);                  // [w + extra, n_tokens]
    cb(sel, "mtp_share_sel", -1);

    res->add_input(std::move(inp));
    return ggml_reshape_4d(ctx0, sel, w + QWEN4EXP_SHARE_EXTRA, n_tokens, 1, 1);
}

void llama_model_qwen4exp::graph::build_qsa_store_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;

    if (qsa_k_inp == nullptr) {
        auto inp = std::make_unique<llm_graph_input_qsa_k>(mctx_hyb);
        inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa_k_inp = inp.get();
        res->add_input(std::move(inp));
    }

    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, qsa_k_inp->k_idxs, il));
}

static void qwen4exp_append_strip(ggml_context * ctx, ggml_cgraph * graph,
        std::vector<ggml_tensor *> & chunks, ggml_tensor * value) {
    chunks.push_back(value);
    while (chunks.size()>1 && chunks.back()->ne[1]==chunks[chunks.size()-2]->ne[1]) {
        auto * right=chunks.back();chunks.pop_back();
        auto * left=chunks.back();chunks.pop_back();
        auto * joined=ggml_concat(ctx,left,right,1);
        chunks.push_back(joined);
        ggml_build_forward_expand(graph,joined);
    }
    ggml_build_forward_expand(graph,chunks.back());
}

static ggml_tensor * qwen4exp_finish_strips(ggml_context * ctx, ggml_cgraph * graph,
        std::vector<ggml_tensor *> & chunks) {
    while (chunks.size()>1) {
        auto * right=chunks.back();chunks.pop_back();
        auto * left=chunks.back();chunks.pop_back();
        auto * joined=ggml_concat(ctx,left,right,1);
        chunks.push_back(joined);
        ggml_build_forward_expand(graph,joined);
    }
    return chunks.front();
}

static int64_t qwen4exp_query_strip(int64_t n_tokens, int64_t n_stream) {
    const char * value = getenv("LLAMA_QSA_QUERY_STRIP");
    const int64_t requested = value ? strtoll(value, nullptr, 10) : 0;
    return n_stream == 1 && requested > 0 ? std::min(n_tokens, requested) : n_tokens;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // strixllama: a compact ubatch's block list covers only its own sequences when the memory allows it
    // (llama_memory_hybrid_idx_context::qsa_active_blocks); otherwise every cell of the pool
    const bool qsa_compact = qwen4exp_use_block_selection(blk_bias,n_stream,r,n_kv,ubatch,cparams,hparams) &&
        hparams.n_swa==0 && mctx_hyb->qsa_scalar_visibility(ubatch) && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA");
    const int64_t n_active = qsa_compact ? mctx_hyb->qsa_active_blocks(ubatch, r, hparams.indexer_top_k/r) : 0;
    const int64_t n_blocks = n_active > 0 ? n_active : (n_kv + r - 1)/r;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        const bool scalar = qwen4exp_use_block_selection(blk_bias,n_stream,r,n_kv,ubatch,cparams,hparams) &&
            hparams.n_swa==0 && mctx_hyb->qsa_scalar_visibility(ubatch);
        qsa->compact = scalar && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA");
        qsa->active_only = n_active > 0;
        // strixllama: only the qsa3 kernel honours the indices without a mask, and it declines batches under
        // 128 queries. On HIP the generic flash-attention kernels ignore the indices (sparse gather is
        // CUDA-only), so a maskless op there runs dense attention over the whole batch with no causal
        // mask: every query of a multi-token decode batch sees the tokens after it. That is exactly the
        // MTP verification batch (2-4 tokens), whose target logits then see the drafts they are meant to
        // check. Keep the mask for anything the qsa3 kernel will not take.
        qsa->maskless = scalar && qwen4exp_qsa_maskless_ok(n_kv, n_tps);
        qsa->score_strip=qwen4exp_query_strip(n_tps,n_stream);
        qsa->score_key_limits=qwen4exp_score_key_limits(mctx_hyb,ubatch,n_blocks,qsa->score_strip,r,
                hparams.indexer_top_k/r,qsa->compact,qsa->active_only);
        qsa->bias = qsa->compact ? ggml_new_tensor_1d(ctx0,GGML_TYPE_I32,n_blocks+n_tps) :
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);
        if (qwen4exp_use_block_selection(blk_bias,n_stream,r,n_kv,ubatch,cparams,hparams)) {
            GGML_ASSERT(hparams.indexer_top_k % r == 0);
            qsa->tail_idxs=ggml_new_tensor_3d(ctx0,GGML_TYPE_I32,r-1,n_tps,n_stream);
            ggml_set_input(qsa->tail_idxs);
        }
        // strixllama: a compact ubatch of several sequences (unified cache, several slots decoding or
        // prefilling in one step) carries the block ownership as membership inputs; this is what keeps
        // the mixed-sequence graph - the reserve graph included - off the dense n_kv x n_tokens mask
        if (qsa->compact && ubatch.n_seqs_unq > 1) {
            qsa->seq_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ubatch.n_seqs_unq, n_blocks);
            qsa->seq_tok = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ubatch.n_seqs_unq, n_tps);
            ggml_set_input(qsa->seq_blk);
            ggml_set_input(qsa->seq_tok);
        }
        // strixllama: block-key cache inputs (llama_memory_hybrid_idx::qsa_kb_inputs); single stream, 1-D
        // positions only. qwen4exp_pos_scalar covers this ubatch; kb_pos_dup covers the cells already
        // in the cache, because an image written earlier makes set_input_qsa rank cells even for a
        // later text ubatch, and the cache cannot track ranked cells (it asserts !ranked).
        qsa->kb_dup = mctx_hyb->kb_pos_dup();
        if (mctx_hyb->get_kb(il) != nullptr && n_stream == 1 && qwen4exp_pos_scalar(ubatch) &&
                !qsa->kb_dup) {
            // strixllama: a ubatch of several sequences can complete up to two blocks per sequence, and each can bring
            // the blocks of up to kb_pending_max positions a graph without the indexer left (kb_note_unrefreshed)
            const int64_t dirty_max = qwen4exp_kb_dirty_max(n_tps, r, ubatch.n_seqs_unq);
            // an input no node reads is never allocated, so a full-rebuild graph gets only the rows tensor
            qsa->kb_full        = mctx_hyb->kb_needs_full(ubatch);
            qsa->kb_bid_rows    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blocks);
            if (!qsa->kb_full) {
                qsa->kb_dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*dirty_max, 1);
                qsa->kb_dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*dirty_max);
                qsa->kb_dirty_dst   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, dirty_max);
            }
            ggml_set_input(qsa->kb_bid_rows);
            if (!qsa->kb_full) {
                ggml_set_input(qsa->kb_dirty_cells);
                ggml_set_input(qsa->kb_dirty_pos);
                ggml_set_input(qsa->kb_dirty_dst);
            }
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // r member keys -> one mean-pooled, normed, rotated key per block (rope wants [n_dims, n_head, n_tokens])
    // the block count along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    auto build_block_keys = [&](ggml_tensor * cells_idx, ggml_tensor * pos_idx, int64_t nb_out) -> ggml_tensor * {
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, cells_idx);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, nb_out, n_stream);
        ggml_tensor * pk = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, members, idx_dim, nb_out, n_stream,
                            members->nb[2], members->nb[3], i*members->nb[1]));
            pk = pk ? ggml_add(ctx0, pk, slice) : slice;
        }
        pk = ggml_scale(ctx0, pk, 1.0f/(float) r);
        pk = ggml_reshape_3d(ctx0, pk, idx_dim, nb_out*n_stream, 1);
        pk = build_norm(pk, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
        pk = ggml_reshape_3d(ctx0, pk, idx_dim, 1, nb_out*n_stream);
        pk = ggml_rope_multi(ctx0, pk, pos_idx, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        return ggml_reshape_3d(ctx0, pk, idx_dim, nb_out, n_stream);
    };

    // strixllama: with the block-key cache only the blocks this ubatch completes are built and stored; the
    // scorer gathers every enumerated block's key from the cache. Without it (or on the first graph after
    // positions moved) every block is rebuilt from the raw keys and, if the cache exists, written to it.
    ggml_tensor * kb_raw = inp->kb_bid_rows ? mctx_hyb->get_kb(il) : nullptr;
    // wrap the externally owned cache in a ctx0 view, as the KV cache does through get_k()
    ggml_tensor * kb = kb_raw ? ggml_view_2d(ctx0, kb_raw, kb_raw->ne[0], kb_raw->ne[1], kb_raw->nb[1], 0) : nullptr;
    ggml_tensor * pooled = nullptr;
    if (kb && !inp->kb_full) {
        const int64_t dirty_max = inp->kb_dirty_dst->ne[0];
        ggml_tensor * fresh = build_block_keys(inp->kb_dirty_cells, inp->kb_dirty_pos, dirty_max);
        cb(fresh, "indexer_kb_fresh", il);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, kb, ggml_reshape_2d(ctx0, fresh, idx_dim, dirty_max), inp->kb_dirty_dst));
        pooled = ggml_get_rows(ctx0, kb, inp->kb_bid_rows);
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    } else {
        pooled = build_block_keys(inp->blk_cells, inp->blk_pos, n_blocks);
        if (kb) {
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, kb, ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks), inp->kb_bid_rows));
            // score with the keys as the cache holds them (F16), as every later graph does: a rebuild - the first
            // graph of a conversation, one after a restore or a rewind - then selects what the cache would have
            pooled = ggml_get_rows(ctx0, kb, inp->kb_bid_rows);
            pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        } else {
            // without the cache - switched off, several streams, an image ubatch, or any text after an image
            // (kb_pos_dup) - the keys are rounded to F16 too: the block choice, and the text, came out different from
            // a fresh server's once the server had seen one image (issue #16)
            pooled = ggml_cast(ctx0, ggml_cast(ctx0, pooled, GGML_TYPE_F16), GGML_TYPE_F32);
        }
    }
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    const int64_t strip = qwen4exp_query_strip(n_tps, n_stream);
    const char * fused_score_env = getenv("LLAMA_QSA_SCORE_WMMA");
    const bool use_fused_score = fused_score_env && atoi(fused_score_env) != 0 &&
        n_tps >= 128 && idx_dim == 128 && n_idx_h == 4;
    ggml_tensor * shared_weights = nullptr;
    ggml_tensor * shared_zero_mask = nullptr;
    if (use_fused_score) {
        const std::string weights_name = format("qsa_score_weights_%lld_%lld_%lld",
                (long long)n_idx_h, (long long)strip, (long long)n_stream);
        const std::string mask_name = format("qsa_score_zero_%lld_%lld_%lld",
                (long long)n_blocks, (long long)strip, (long long)n_stream);
        shared_weights = ggml_get_tensor(ctx0, weights_name.c_str());
        shared_zero_mask = ggml_get_tensor(ctx0, mask_name.c_str());
        if (!shared_weights) {
            shared_weights = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                    n_idx_h, strip, 1, n_stream), 1.0f);
            ggml_set_name(shared_weights, weights_name.c_str());
        }
        if (!shared_zero_mask) {
            shared_zero_mask = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F16,
                    n_blocks, strip, 1, n_stream), 0.0f);
            ggml_set_name(shared_zero_mask, mask_name.c_str());
        }
    }
    std::vector<ggml_tensor *> selected;
    for (int64_t first = 0; first < n_tps; first += strip) {
        const int64_t n_query = std::min(strip, n_tps - first);
        // [QSA_SCORE_BOUNDS] trim the scorer to the blocks this strip can actually see.
        // The fused WMMA scorer sizes its shared zero mask from n_blocks, so it opts out.
        const int64_t score_blocks = (use_fused_score || inp->score_key_limits.empty())
                ? n_blocks : inp->score_key_limits[first/strip];
        ggml_tensor * score_keys = pooled;
        if (score_blocks < n_blocks) {
            score_keys = ggml_view_3d(ctx0, pooled, idx_dim, score_blocks, n_stream, pooled->nb[1], pooled->nb[2], 0);
            cb(score_keys, "indexer_k_bounded", il);
        }
        ggml_tensor * bias = inp->compact ? nullptr : ggml_view_3d(ctx0, inp->bias, inp->bias->ne[0], n_query, n_stream,
                inp->bias->nb[1], inp->bias->nb[2], first*inp->bias->nb[1]);
        ggml_tensor * query_mask = kq_mask == nullptr ? nullptr : ggml_view_4d(ctx0, kq_mask,
                n_kv, n_query, 1, n_stream, kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], first*kq_mask->nb[1]);

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * score;
        if (use_fused_score) {
            ggml_tensor * query = ggml_view_4d(ctx0, q, idx_dim, n_idx_h, n_query, n_stream,
                    q->nb[1], q->nb[2], q->nb[2]*n_tps, first*q->nb[2]);
            ggml_tensor * key = ggml_reshape_4d(ctx0, pooled, idx_dim, 1, n_blocks, n_stream);
            ggml_tensor * weights = n_query == strip ? shared_weights : ggml_view_4d(ctx0, shared_weights,
                    n_idx_h, n_query, 1, n_stream, shared_weights->nb[1], shared_weights->nb[2], shared_weights->nb[3], 0);
            ggml_tensor * zero_mask = n_query == strip ? shared_zero_mask : ggml_view_4d(ctx0, shared_zero_mask,
                    n_blocks, n_query, 1, n_stream, shared_zero_mask->nb[1], shared_zero_mask->nb[2], shared_zero_mask->nb[3], 0);
            score = ggml_reshape_3d(ctx0, ggml_lightning_indexer(ctx0, query, key, weights, zero_mask),
                    n_blocks, n_query, n_stream);
        } else {
        score = ggml_mul_mat(ctx0, score_keys,
                ggml_view_3d(ctx0, q, idx_dim, n_idx_h*n_query, n_stream, q->nb[1], q->nb[2]*n_tps, first*q->nb[2]));
        score = ggml_reshape_4d(ctx0, score, score_blocks, n_idx_h, n_query, n_stream);
        score = ggml_relu(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, score_blocks, n_query, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        score = summed;
        }
        cb(score, "indexer_score", il);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (inp->compact) {
            score = qwen4exp_apply_compact_visibility(ctx0,score,inp->bias,n_blocks,first,n_query,inp->seq_blk,inp->seq_tok);
        } else if (blk_bias) {
            score = ggml_add(ctx0, score, bias);
        }

        if (inp->tail_idxs) {
            ggml_tensor * tail=ggml_view_4d(ctx0,inp->tail_idxs,r-1,n_query,1,n_stream,
                    inp->tail_idxs->nb[1],inp->tail_idxs->nb[2],inp->tail_idxs->nb[2],first*inp->tail_idxs->nb[1]);
            ggml_tensor * block_cells = score_blocks < n_blocks ?
                ggml_view_2d(ctx0, inp->blk_cells, r*score_blocks, n_stream, inp->blk_cells->nb[1], 0) : inp->blk_cells;
            ggml_tensor * top_k=qwen4exp_select_complete_blocks(ctx0,score,block_cells,tail,
                    hparams.indexer_top_k/r,r);
            cb(top_k,"indexer_top_k",il);
            qwen4exp_append_strip(ctx0,gf,selected,top_k);
            continue;
        }

        // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            // flash attention keeps the mask in f16; the scores are f32
            ggml_tensor * mask = query_mask->type == GGML_TYPE_F32 ? query_mask : ggml_cast(ctx0, query_mask, GGML_TYPE_F32);
            expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_query, n_stream));
        } else {
            expanded = ggml_add(ctx0, expanded, bias);
        }
        cb(expanded, "indexer_score_tokens", il);

        // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
        const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

        ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

        // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
        top_k = ggml_reshape_4d(ctx0, top_k, width, n_query, 1, n_stream);
        cb(top_k, "indexer_top_k", il);
        qwen4exp_append_strip(ctx0,gf,selected,top_k);
    }
    ggml_tensor * top_k = qwen4exp_finish_strips(ctx0,gf,selected);

    return top_k;
}

// strixllama: decode-side sparse attention by gathering.
//
// The qsa3 kernel is the only one that honours the block indices, and it needs the packed K/V layout
// that qwen4exp builds only for batches of >= 128 tokens; every other kernel ignores src[5] and reads
// the whole cache. So a single decoded token attends densely over the entire context - 15.6 ms of a
// 58 ms token at 97K (results/hip-rocm101-20260917.json, decode_node_timing_97k_20260918).
//
// For a decode-sized batch the selection is small and fixed (indexer_top_k + ratio - 1 = 2051 cells),
// so it is cheaper to materialise it: gather those cells out of the cache into a compact K/V and run
// the ordinary flash-attention kernel over 2051 keys instead of 97512. The selection carries -1 for
// "no cell" (block-graph.inc), which becomes a -inf mask entry, and the gather length is padded to a
// multiple of FATTN_KQ_STRIDE with the same -inf so every kernel variant can take it.
//
// All queries go through one gather and one flash-attention call: the queries sit on the kernel's
// sequence axis (ne[3]), each with its own K/V slab and mask row, so the node count does not grow
// with the batch. That is what lets a ubatch that serves several slots (4 streams x 4 verify tokens
// = 16 queries) stay on this path instead of falling back to dense attention over the whole pool.
// The gathered rows stay f16: ggml_get_rows only offers an f32 result, which doubled the traffic
// (read f16, write f32, read f32, write f16) - the CUDA kernel casts to whatever the destination
// is, so the node is built here with an f16 destination.
//
// This makes decode sparse, which is what the model intends and what prefill already does; it is not
// bit-identical to the dense decode it replaces.
static ggml_tensor * qwen4exp_get_rows_f16(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    // strixllama: a Q8_0 cache is dequantized by the same kernel on the way out
    GGML_ASSERT((a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_Q8_0) && b->type == GGML_TYPE_I32 && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1);
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, a->ne[0], b->ne[0]);
    result->op     = GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

static ggml_tensor * qwen4exp_gather_attn(ggml_context * ctx0, ggml_tensor * q, ggml_tensor * k,
        ggml_tensor * v, ggml_tensor * top_k, int64_t n_query, float kq_scale) {
    const int64_t head_k  = k->ne[0];
    const int64_t n_head_kv = k->ne[1];
    const int64_t width   = top_k->ne[0];
    const int64_t n_sel   = (width + 255) & ~255;   // FATTN_KQ_STRIDE

    GGML_ASSERT(ggml_is_contiguous(top_k) && top_k->ne[1] >= n_query);

    // [head_k, n_head_kv, n_kv] -> [head_k*n_head_kv, n_kv], so a cell is one row
    ggml_tensor * k2d = ggml_view_2d(ctx0, k, head_k*n_head_kv, k->ne[2], k->nb[2], 0);
    ggml_tensor * v2d = ggml_view_2d(ctx0, v, v->ne[0]*v->ne[1], v->ne[2], v->nb[2], 0);

    // every query's selection at once: [width, n_query] -> [n_sel, n_query]
    ggml_tensor * idx = ggml_view_2d(ctx0, top_k, width, n_query, top_k->nb[1], 0);

    // shift by one so "no cell" (-1) becomes 0 and the zero padding means the same thing
    ggml_tensor * shifted = ggml_pad(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), 1.0f, 1.0f),
            (int) (n_sel - width), 0, 0, 0);
    ggml_tensor * valid = ggml_step(ctx0, shifted);
    ggml_tensor * rows  = ggml_cast(ctx0, ggml_relu(ctx0, ggml_scale_bias(ctx0, shifted, 1.0f, -1.0f)), GGML_TYPE_I32);
    rows = ggml_reshape_1d(ctx0, rows, n_sel*n_query);
    ggml_tensor * mask  = ggml_cast(ctx0, ggml_log(ctx0, valid), GGML_TYPE_F16);   // 0 where selected, -inf elsewhere
    mask = ggml_cont(ctx0, ggml_reshape_4d(ctx0, mask, n_sel, 1, 1, n_query));

    // strixllama: the cache is read in place through the cell list (ggml_flash_attn_ext_gather): the same numbers without
    // the gathered f16 copy - its write and the attention's second read. Eight conversations at 20K spent ~3.0 ms of a
    // step on the gathers and ~3.2 on the attention (STRIX_REPEAT, 0.3.5); fused, ~2.2 ms. STRIX_FA_GATHER=0 keeps the copy
    static const bool fused = [] {
        const char * e = getenv("STRIX_FA_GATHER");
        return e == nullptr || atoi(e) != 0;
    }();
    if (fused && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && v->type == k->type && head_k == 256 &&
            v->ne[0] == 256 && k->ne[3] == 1 && v->ne[3] == 1 && q->ne[1] % n_head_kv == 0) {
        ggml_tensor * q4 = ggml_permute(ctx0, ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], 1, n_query,
                q->nb[1], q->nb[2], q->nb[2], 0), 0, 2, 1, 3);
        return ggml_flash_attn_ext_gather(ctx0, q4, k, v, mask, rows, kq_scale);   // [head_v, n_head_q, 1, n_query]
    }

    // one K/V slab per query on the sequence axis: [head, n_sel, n_head_kv, n_query]
    ggml_tensor * kg = qwen4exp_get_rows_f16(ctx0, k2d, rows);
    ggml_tensor * vg = qwen4exp_get_rows_f16(ctx0, v2d, rows);
    kg = ggml_permute(ctx0, ggml_reshape_4d(ctx0, kg, head_k,    n_head_kv, n_sel, n_query), 0, 2, 1, 3);
    vg = ggml_permute(ctx0, ggml_reshape_4d(ctx0, vg, v->ne[0], n_head_kv, n_sel, n_query), 0, 2, 1, 3);

    // q is [head_k, n_head_q, n_query]; one query per sequence, laid out as the kernel wants
    ggml_tensor * q4 = ggml_permute(ctx0, ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], 1, n_query,
            q->nb[1], q->nb[2], q->nb[2], 0), 0, 2, 1, 3);

    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q4, kg, vg, mask, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    return cur;   // [head_v, n_head_q, 1, n_query]
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * mask_all = inp->get_kq_mask();
    ggml_tensor * indices_all = top_k;
    const int64_t n_stream = mask_all->ne[3];
    const int64_t n_tps = mask_all->ne[1];
    const auto shared_qsa=qsa_inps.find((uint32_t)hparams.dsv4_compress_ratios[il]);
    const char * direct_option=getenv("LLAMA_QSA_DIRECT_INDICES");
    const bool layout_prefill=n_tps>=128 && n_stream==1 && direct_option && atoi(direct_option)!=0 &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias==0.0f && !hparams.attn_soft_cap &&
        shared_qsa!=qsa_inps.end() && shared_qsa->second->maskless;
    const char * whole_option=getenv("LLAMA_QSA_WHOLE_ATTN");
    const int64_t strip=layout_prefill && whole_option && atoi(whole_option)!=0 ? n_tps : qwen4exp_query_strip(n_tps,n_stream);
    ggml_tensor * packed_keys=nullptr;
    const char * pack_option=getenv("LLAMA_QSA_PACK_KEYS");
    // strixllama: the sparse prefill kernel reads K and V only through these packed f16 layouts, so a Q8_0 cache is
    // dequantized into them here and the kernel does not change
    auto kv_for_packing = [&](ggml_tensor * t) {
        return t->type == GGML_TYPE_Q8_0 ? ggml_cast(ctx0, t, GGML_TYPE_F16) : t;
    };
    // strixllama: GGML_OP_QSA_PACK writes the packed layouts from the cache view itself, dequantizing a Q8_0 cache as
    // the cast did: no f16 copy of the cache and no CONT of its permuted view, which the compute buffer held for
    // the whole pool (~2 KiB a KV cell). STRIX_QSA_PACK_OP=0 builds the old nodes.
    static const bool pack_op = !getenv("STRIX_QSA_PACK_OP") || atoi(getenv("STRIX_QSA_PACK_OP")) != 0;
    auto pack_direct = [&](const ggml_tensor * c) {
        return pack_op && (c->type == GGML_TYPE_F16 || c->type == GGML_TYPE_Q8_0) &&
            c->ne[0] == 256 && c->ne[2] % 4 == 0 && c->ne[3] == 1 && c->nb[0] == ggml_type_size(c->type);
    };
    if (layout_prefill && pack_option && atoi(pack_option)!=0) {
        ggml_tensor * k_cache = mctx_cur->get_k(ctx0, il);
        if (pack_direct(k_cache)) {
            packed_keys = ggml_qsa_pack(ctx0, k_cache, false);
            ggml_set_name(packed_keys, "qsa_packed_keys");
            ggml_build_forward_expand(gf, packed_keys);
        } else {
        auto * original_keys=ggml_permute(ctx0,kv_for_packing(k_cache),0,2,1,3);
        if (original_keys->type==GGML_TYPE_F16 && original_keys->ne[0]==256 && original_keys->ne[1]%4==0 && original_keys->ne[3]==1) {
            packed_keys=qsa_pack_keys(ctx0,original_keys);
            ggml_build_forward_expand(gf,packed_keys);
        }
        }
    }
    // QSA attention v2 (qsa-attn/qsa2.cu) consumes V^T fragments straight from a [4 keys][256 dims] block
    // layout; build it once per graph next to the packed keys when requested.
    ggml_tensor * packed_values=nullptr;
    const char * packv_option=getenv("LLAMA_QSA_PACK_VALUES");
    if (layout_prefill && packv_option && atoi(packv_option)!=0) {
        ggml_tensor * v_cache = mctx_cur->get_v(ctx0, il);
        if (pack_direct(v_cache)) {
            packed_values = ggml_qsa_pack(ctx0, v_cache, true);
            ggml_set_name(packed_values, "qsa_packed_values");
            ggml_build_forward_expand(gf, packed_values);
        } else {
        auto * original_values=ggml_permute(ctx0,kv_for_packing(v_cache),0,2,1,3);
        if (original_values->type==GGML_TYPE_F16 && original_values->ne[0]==256 && original_values->ne[1]%4==0 && original_values->ne[3]==1) {
            packed_values=qsa_pack_values(ctx0,original_values);
            ggml_build_forward_expand(gf,packed_values);
        }
        }
    }
    std::vector<ggml_tensor *> output;
    for (int64_t first = 0; first < n_tps; first += strip) {
        const int64_t n_query = std::min(strip, n_tps - first);
        ggml_tensor * kq_mask = ggml_view_4d(ctx0, mask_all, mask_all->ne[0], n_query, 1, n_stream,
                mask_all->nb[1], mask_all->nb[2], mask_all->nb[3], first*mask_all->nb[1]);
        ggml_tensor * top_k = ggml_view_4d(ctx0, indices_all, indices_all->ne[0], n_query, 1, n_stream,
                indices_all->nb[1], indices_all->nb[2], indices_all->nb[3], first*indices_all->nb[1]);

        const char * direct_env = getenv("LLAMA_QSA_DIRECT_INDICES");
        const bool direct_indices = direct_env && atoi(direct_env) != 0 && !qwen4exp_qsa_between_kernels(n_tps) &&
            n_stream == 1 && cparams.flash_attn && cparams.offload_kqv &&
            hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap;
        ggml_tensor * kq_mask_top_k = kq_mask;
        if (!direct_indices) {
        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        // reshape KQ mask into tensor with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
        // this will be our source of zero values for unmasking top k mask elements
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        // modify KQ mask by unmasking elements that are in top_k indices
        // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
        kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // reshape to restore the original shape of KQ mask:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // combine with the original kq mask
        kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

        }

        ggml_tensor * q = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], n_query,
                q_cur->nb[1], q_cur->nb[2], first*q_cur->nb[2]);
        if (n_stream != 1) { q = q_cur; }
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);

        // TODO: enable sparse attention when we are ready
        // ref: https://github.com/ggml-org/llama.cpp/pull/27970
        //ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
        const char * sparse_env = getenv("LLAMA_QSA_SPARSE");
        const int64_t n_kv_max = sparse_env && atoi(sparse_env) != 0 ? top_k->ne[0] : 0;
        // strixllama: decode-sized batches gather the selected cells instead of reading the whole cache
        static const bool gather_on = []() {
            const char * e = getenv("LLAMA_QSA_DECODE_GATHER");
            return e != nullptr && atoi(e) != 0;
        }();
        const int64_t gather_max_t = qwen4exp_gather_max_t();
        const bool gather = gather_on && n_query <= gather_max_t && n_stream == 1 &&
            cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap &&
            (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0) &&
            v->nb[1] <= v->nb[2] &&
            q->ne[0] == 256 &&
            // only worth it when the selection is a real reduction: gathering 2304 of 2329 cells costs
            // more than reading them in place (measured 42.9 vs 41.1 ms/token at 2.3K context)
            2*(((top_k->ne[0] + 255) & ~255)) <= k->ne[2];

        ggml_tensor * cur;
        if (gather) {
            ggml_tensor * idx = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
            cur = qwen4exp_gather_attn(ctx0, q, k, v, idx, n_query, kq_scale);
            // same layout the other branches hand to qwen4exp_append_strip: [head_v*n_head_q, n_query]
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            ggml_build_forward_expand(gf, cur);
        } else if (direct_indices) {
            GGML_ASSERT(q->ne[0] == 256 && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
                                           (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0));
            const bool v_trans = v->nb[1] > v->nb[2];
            ggml_tensor * q_view = ggml_permute(ctx0, q, 0, 2, 1, 3);
            ggml_tensor * k_view = ggml_permute(ctx0, k, 0, 2, 1, 3);
            ggml_tensor * v_view = ggml_permute(ctx0, v, 0, 2, 1, 3);
            if (v_trans) { v_view = ggml_transpose(ctx0, v_view); }
            const auto qsa_it=qsa_inps.find((uint32_t)hparams.dsv4_compress_ratios[il]);
            const bool maskless=qsa_it!=qsa_inps.end() && qsa_it->second->maskless;
            ggml_tensor * mask = maskless ? nullptr : (ggml_is_contiguous(kq_mask) ? kq_mask : ggml_cont(ctx0, kq_mask));
            ggml_tensor * indices = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
            GGML_ASSERT(indices->ne[1] == kq_mask->ne[1] && indices->ne[3] == kq_mask->ne[3]);
            cur = ggml_flash_attn_ext(ctx0, q_view, k_view, v_view, mask, kq_scale, 0.0f, 0.0f);
            cur->src[5] = indices;
            cur->src[6] = packed_keys;
            cur->src[7] = packed_values;
            ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(indices->ne[0]));
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            ggml_build_forward_expand(gf, cur);
        } else {
            cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, n_kv_max, kq_scale, il);
        }
        qwen4exp_append_strip(ctx0,gf,output,cur);
    }
    ggml_tensor * cur = qwen4exp_finish_strips(ctx0,gf,output);

    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = nullptr;
    if (qsa) {
        const int64_t r     = hparams.dsv4_compress_ratios[il];
        const int64_t n_kv  = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width = (int64_t) hparams.indexer_top_k + r - 1;
        static const bool shortcut = [] {
            const char * env = getenv("LLAMA_QSA_DENSE_SHORTCUT");
            return env == nullptr || atoi(env) != 0;
        }();
        if (shortcut && n_kv <= width) {
            build_qsa_store_k(mctx_hyb, cur, il);
        } else {
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il);
        }
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

// strixllama: the sequence sets of a ubatch in groups of one token count - split_ragged puts sets of one count next to
// each other (llama_ubatch::ragged): group g is n_seqs sets from set s0, n_tok tokens each, its tokens from t0. An
// equal-length ubatch is one group
struct qwen4exp_seq_group {
    int64_t s0;
    int64_t n_seqs;
    int64_t n_tok;
    int64_t t0;
};

static std::vector<qwen4exp_seq_group> qwen4exp_seq_groups(const llama_ubatch & ub) {
    std::vector<qwen4exp_seq_group> res;
    for (uint32_t s = 0; s < ub.n_seqs; ++s) {
        const int64_t n  = ub.ntok(s);
        const int64_t t0 = ub.tok0(s);
        if (!res.empty() && res.back().n_tok == n && res.back().t0 + res.back().n_seqs * n == t0) {
            res.back().n_seqs++;
        } else {
            res.push_back({ (int64_t) s, 1, n, t0 });
        }
    }
    return res;
}

// strixllama: the groups' outputs back in one tensor, a column of `rows` values per token, in token order.
// split_ragged puts the longest group first; the short ones are joined first, so the long one is copied once
static ggml_tensor * qwen4exp_concat_groups(ggml_context * ctx, const std::vector<ggml_tensor *> & parts, int64_t rows) {
    std::vector<ggml_tensor *> cols;
    for (ggml_tensor * p : parts) {
        cols.push_back(ggml_reshape_2d(ctx, p, rows, ggml_nelements(p) / rows));
    }
    ggml_tensor * rest = nullptr;
    for (size_t k = 1; k < cols.size(); ++k) {
        rest = rest ? ggml_concat(ctx, rest, cols[k], 1) : cols[k];
    }
    return rest ? ggml_concat(ctx, cols[0], rest, 1) : cols[0];
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;

    // strixllama: a ragged ubatch (llama_ubatch::ragged) builds the per-sequence part - the conv, the net and the
    // states they read and write - once per group of sets with one token count (qwen4exp_seq_groups), and the per-token
    // rest once, as a single set of all the tokens. An equal-length ubatch is one group and builds the graph it did
    const bool    ragged       = ubatch.ragged();
    const int64_t n_seq_tokens = ragged ? (int64_t) ubatch.n_tokens : (int64_t) ubatch.n_seq_tokens;
    const int64_t n_seqs_tok   = ragged ? 1 : n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs_tok);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    // strixllama: the two narrow projections of the same input side by side in the graph, which ggml-cuda then runs as
    // one pass over the input for a prompt-sized batch (the narrow pair, ggml_cuda_mmb_f32_narrow_pair). Graph order only
    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, alpha);

    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs_tok);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs_tok);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs_tok);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    // strixllama: deferred rollback (llm_graph_input_rs::lazy_mode): a short batch runs the net that records instead
    // of snapshotting and reads its states from their rows itself; any other first replays the records the states
    // still carry, in place, so the gather and the plain net see them as they expect. A ragged ubatch is never short
    const int lazy_mode = inp->lazy_mode;
    GGML_ASSERT(!ragged || lazy_mode != 1);
    ggml_tensor * states = nullptr;

    std::vector<ggml_tensor *> outputs;
    for (const auto & g : qwen4exp_seq_groups(ubatch)) {
        const int64_t n_grp_tok  = g.n_tok;
        const int64_t n_grp_seqs = g.n_seqs;

        // the group's tokens, the columns [t0, t0 + n_tok*n_seqs) - the whole tensors when it is the only group
        ggml_tensor * x_g    = qkv_mixed;
        ggml_tensor * beta_g = beta;
        ggml_tensor * gate_g = gate;
        if (ragged) {
            x_g    = ggml_view_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_grp_tok, n_grp_seqs,
                    qkv_mixed->nb[1], qkv_mixed->nb[1] * n_grp_tok, qkv_mixed->nb[1] * g.t0);
            beta_g = ggml_view_4d(ctx0, beta, 1, num_v_heads, n_grp_tok, n_grp_seqs,
                    beta->nb[1], beta->nb[2], beta->nb[2] * n_grp_tok, beta->nb[2] * g.t0);
            gate_g = ggml_view_4d(ctx0, gate, 1, num_v_heads, n_grp_tok, n_grp_seqs,
                    gate->nb[1], gate->nb[2], gate->nb[2] * n_grp_tok, gate->nb[2] * g.t0);
        }

        ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, x_g,
                conv_kernel_size - 1, conv_channels, il, g.s0, ragged ? n_grp_seqs : -1);

        ggml_tensor * state = nullptr;
        if (lazy_mode != 1) {
            if (states == nullptr) {
                ggml_tensor * ssm_src = ssm_states_all;
                if (lazy_mode == 2) {
                    ssm_src = ggml_gated_delta_net_replay(ctx0, ssm_states_all, mctx_cur->get_rec_l(il), inp->lazy_info,
                            (int) head_v_dim, (int) num_v_heads, (int) num_k_heads);
                }
                states = build_rs(inp, ssm_src, hparams.n_embd_s(), n_seqs);
            }
            state = ragged ? ggml_view_2d(ctx0, states, states->ne[0], n_grp_seqs, states->nb[1], states->nb[1] * g.s0)
                           : states;
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_grp_seqs);
            cb(state, "state_predelta", il);
        }

        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);

        ggml_tensor * conv_qkv_mix = conv_output_silu;

        int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

        // Extract the convolved Q, K, V from conv_output
        ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_grp_tok, n_grp_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_grp_tok,
                0);

        ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_grp_tok, n_grp_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim),
                nb1_qkv,
                nb1_qkv * n_grp_tok,
                head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

        ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_grp_tok, n_grp_seqs,
                ggml_row_size(conv_qkv_mix->type, head_v_dim),
                nb1_qkv,
                nb1_qkv * n_grp_tok,
                ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);

        const float eps_norm = hparams.f_norm_rms_eps;

        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

        // strixllama: ggml-cuda runs the conv - and these two norms - straight from x_g and the gathered state rows at the
        // conv's node (gdn-conv.cu), and the allocator frees both after the concat, the last node reading them: the SiLU's
        // output could land on them while that pass reads them. The norms' outputs take the state rows (3 x channels) as
        // an extra source, which no backend reads and which keeps them allocated past the SiLU's. x_g kept too raised the
        // compute buffer by 50 MB: gdn-conv.cu copies the halos when the SiLU's output is x_g's block, and falls back when
        // it is elsewhere on it
        if (q_conv->op == GGML_OP_SCALE && k_conv->op == GGML_OP_SCALE && conv_input->op == GGML_OP_CONCAT) {
            for (ggml_tensor * t : { q_conv, k_conv }) {
                GGML_ASSERT(t->src[1] == nullptr);
                t->src[1] = conv_input->src[0];
            }
        }

        // repeat to match shapes when head keys != value keys; unneeded with the fused GDN (and the deferred-rollback
        // net)
        if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch) && lazy_mode != 1) {
            GGML_ASSERT(num_v_heads % num_k_heads == 0);
            q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_grp_tok, n_grp_seqs);
            k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_grp_tok, n_grp_seqs);
        }

        cb(q_conv, "q_conv_predelta", il);
        cb(k_conv, "k_conv_predelta", il);
        cb(v_conv, "v_conv_predelta", il);

        ggml_tensor * output = nullptr;
        if (lazy_mode == 1) {
            output = ggml_gated_delta_net_lazy(ctx0, q_conv, k_conv, v_conv, gate_g, beta_g, ssm_states_all,
                    mctx_cur->get_rec_l(il), inp->lazy_info);
            output = ggml_reshape_4d(ctx0, output, head_v_dim, num_v_heads, n_grp_tok, n_grp_seqs);
            cb(output, "attn_output", il);
        } else {
            output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate_g, beta_g, state, il, g.s0);
        }
        outputs.push_back(output);
    }

    ggml_tensor * output = outputs[0];
    if (ragged) {
        output = qwen4exp_concat_groups(ctx0, outputs, head_v_dim * num_v_heads);
        output = ggml_reshape_4d(ctx0, output, head_v_dim, num_v_heads, n_seq_tokens, n_seqs_tok);
    }

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs_tok);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // strixllama: one column per token, not a batch per sequence (STRIX_GDN_OUT_2D=0: as upstream): with several
    // sequences decoding at once, a 3-D input made the projection a batched matrix-vector product per sequence
    static const bool out_2d = !getenv("STRIX_GDN_OUT_2D") || atoi(getenv("STRIX_GDN_OUT_2D")) != 0;
    ggml_tensor * final_output = out_2d && n_seqs_tok > 1
        ? ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs_tok)
        : ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs_tok);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs_tok);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    const bool has_shexp = model.layers[il].ffn_up_shexp != nullptr;
    ggml_tensor * ffn_shexp   = nullptr;
    ggml_tensor * shared_gate = nullptr;
    auto build_shexp = [&]() {
        ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ggml_build_forward_expand(gf, ffn_shexp);
        ggml_build_forward_expand(gf, shared_gate);
    };
    // strixllama: for a prompt-sized batch the shared expert enters the graph between the routed experts' down projection
    // and their weighted sum, so that its gating and the final add come right after the sum, which then runs them in
    // the same kernel (ggml-cuda, STRIX_SHEXP_TAIL). Graph order only: every value is the same. Before the routed
    // experts it would end the routed input's life early and let their fused gate/up kernel's output take its buffer,
    // which that kernel cannot share. STRIX_SHEXP_MOVE=0: the old order
    static const bool shexp_move = !getenv("STRIX_SHEXP_MOVE") || atoi(getenv("STRIX_SHEXP_MOVE")) != 0;
    if (has_shexp && shexp_move && n_tokens >= 512) {
        moe_before_weighting = build_shexp;
    }

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    moe_before_weighting = nullptr;   // a path of build_moe_ffn without the hook point

    // shared experts, as in the Qwen3Next reference
    if (has_shexp) {
        if (!ffn_shexp) {
            build_shexp();
        }

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        const int64_t n = (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
        return pmodel.ple_reader ? data->ne[1] == n : rows->ne[0] == n;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]
    ggml_tensor * data = nullptr;   // direct mode: staged rows [ple_head_dim, ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<uint8_t> staging;
};

// Prefetch hook: the chunk-boundary stall is the PLE row gather (257k-393k scattered 90-byte reads, ~300 ms with the
// GPU idle). Given the whole batch up front we can compute the same row indices and warm the page cache for them while
// the previous chunk is still on the GPU. Prefetch only warms the file cache; it does not change gathered rows.
// strixllama: `n_skip` leading tokens are context only (the n-gram lookups of the first real token reach back into
// them); with `pregather` the rows are gathered and dequantized for the gather of exactly that batch to take, rather
// than only warmed in the file cache
// the table rows the gather of these tokens reads, for the tokens after the first n_skip (context only)
static std::vector<int32_t> qwen4exp_ple_rows(const llama_hparams & hp, const std::vector<llama_token> & toks, int64_t n_skip) {
    const int64_t n_gram = hp.ple_ngram_size, n_heads = hp.ple_n_heads, per_gram = hp.ple_heads_per_ngram;
    const int64_t eos = hp.ple_eos_token_id;
    const int64_t n = (int64_t) toks.size();
    std::vector<int32_t> idx((size_t) n_heads * (n - n_skip));
    std::vector<int64_t> ctx(n_gram);
    for (int64_t i = n_skip; i < n; ++i) {
        ctx[0] = toks[i];
        bool cut = false;
        for (int64_t sft = 1; sft < n_gram; ++sft) {
            const int64_t j = i - sft;                       // contiguous single-sequence prefill
            const llama_token t = (cut || j < 0) ? LLAMA_TOKEN_NULL : toks[j];
            cut = cut || t < 0 || t == eos;
            ctx[sft] = cut ? eos : t;
        }
        for (int64_t g = 2; g <= n_gram; ++g) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < g; ++j) { mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j]; }
            const int64_t base = (g - 2) * per_gram;
            for (int64_t q = 0; q < per_gram; ++q) {
                const int64_t h_i = base + q;
                idx[(size_t) (i - n_skip) * n_heads + h_i] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }
    return idx;
}

// strixllama: the rows of the batch about to run, read at once on a thread of their own (llama_strix_pregather_now):
// a prompt's first batch, which no earlier batch announced, otherwise reads them only after its graph is built
void qwen4exp_ple_pregather_now(const llama_model & model_base, const llama_token * tokens, int32_t n_tokens, int32_t n_skip) {
    if (!tokens || n_tokens - n_skip < 64 || n_skip < 0 || model_base.arch != LLM_ARCH_QWEN4EXP) { return; }
    const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model_base);
    static const bool off = (getenv("LLAMA_PLE_PREFETCH") && atoi(getenv("LLAMA_PLE_PREFETCH")) == 0) ||
                            (getenv("STRIX_PLE_NOW") && atoi(getenv("STRIX_PLE_NOW")) == 0);
    if (!pmodel.ple_reader || off || pmodel.hparams.ple_n_heads <= 0 || pmodel.hparams.ple_ngram_size < 2) { return; }
    pmodel.ple_reader->pregather_now(qwen4exp_ple_rows(pmodel.hparams, std::vector<llama_token>(tokens, tokens + n_tokens), n_skip));
}

void qwen4exp_ple_prefetch(const llama_model & model_base, const llama_token * tokens, int32_t n_tokens,
                           int32_t n_skip, bool pregather) {
    // strixllama: llama_decode calls this for every model; the cast below is only valid for this one
    // strixllama: a pregather serves chunks of any length (the server announces only real ones); a page-cache prefetch
    // pays off only for big batches
    if (!tokens || n_tokens - n_skip < (pregather ? 64 : 4096) || model_base.arch != LLM_ARCH_QWEN4EXP) { return; }
    const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model_base);
    if (!pmodel.ple_reader) { return; }
    static const bool off = getenv("LLAMA_PLE_PREFETCH") && atoi(getenv("LLAMA_PLE_PREFETCH")) == 0;
    if (off) { return; }
    const auto & hp = pmodel.hparams;
    if (hp.ple_n_heads <= 0 || hp.ple_ngram_size < 2) { return; }
    std::vector<llama_token> toks(tokens, tokens + n_tokens);
    auto prefetch = [reader = pmodel.ple_reader, toks = std::move(toks), hp, n_skip, pregather]() {
        const std::vector<int32_t> idx = qwen4exp_ple_rows(hp, toks, n_skip);
        if (pregather) {
            reader->pregather(idx.data(), (int64_t) idx.size());
        } else {
            reader->prefetch(idx.data(), (int64_t) idx.size());
        }
    };
#ifdef _WIN32
    if (pregather) {
        pmodel.ple_reader->defer_prefetch(std::move(prefetch));
    } else {
        pmodel.ple_reader->launch_prefetch(std::move(prefetch));
    }
#else
    std::thread(std::move(prefetch)).detach();
#endif
}

// strixllama: wake the drive the PLE rows are read from ahead of a small batch's gather (llama_lazy_reader::wake);
// llama_context calls it for every model
void qwen4exp_ple_wake(const llama_model & model_base) {
    if (model_base.arch != LLM_ARCH_QWEN4EXP) {
        return;
    }
    const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model_base);
    if (pmodel.ple_reader) {
        pmodel.ple_reader->wake();
    }
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    if (pmodel.ple_reader) {
        // strixllama: rows a caller had gathered while the previous batch was on the GPU (llama_strix_prefetch)
        if (!pmodel.ple_reader->take_pregathered(idx.data(), (int64_t) idx.size(), staging)) {
            staging.resize(idx.size() * pmodel.ple_reader->head_dim * sizeof(float));
            pmodel.ple_reader->gather(idx.data(), (int64_t) idx.size(), (float *) staging.data());
        }
#ifdef _WIN32
        pmodel.ple_reader->launch_deferred();   // the next batch's pregather, now that the drive is free
#endif
        ggml_backend_tensor_set(data, staging.data(), 0, staging.size());
    } else {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    }
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il,
        int64_t              s0,
        int64_t              n_seqs_grp) {
    const auto * mctx_cur = inp->mctx;

    // strixllama: the sequences of one group of a ragged ubatch are the sets from s0 on, in the cells from head + s0
    const auto kv_head = mctx_cur->get_head() + s0;

    const int64_t n_seqs_all = ubatch.n_seqs;
    const int64_t n_seqs     = n_seqs_grp < 0 ? n_seqs_all : n_seqs_grp;
    const int64_t row_total  = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs_all)).first;
    }
    ggml_tensor * rows = it->second;
    if (n_seqs != n_seqs_all) {
        rows = ggml_view_2d(ctx0, rows, row_total, n_seqs, rows->nb[1], rows->nb[1] * s0);
    }

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // strixllama: the copy reads the strided tail itself; a cont first was one more dispatch per slot, 144 of the
        // ~2500 in a 4-token verify pass
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ggml_tensor * emb = nullptr;

    if (static_cast<const llama_model_qwen4exp &>(model).ple_reader) {
        ple_inp->data = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                           hparams.ple_head_dim, n_heads * n_tokens);
        ggml_set_input(ple_inp->data);
        ggml_tensor * data = ple_inp->data;
        res->add_input(std::move(ple_inp));

        // strixllama: the rows of an EXL3 row codec table as ExLlamaV3 reconstructs them, fp16(row + bias[head])
        if (model.per_layer_tok_embd_b) {
            data = llama_weight_aux_row_bias(ctx0, data, model.per_layer_tok_embd_b, n_heads);
        }
        emb = ggml_reshape_2d(ctx0, data, hparams.ple_head_dim * n_heads, n_tokens);
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));

        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
        if (model.per_layer_tok_embd_b) {
            emb = llama_weight_aux_row_bias(ctx0, emb, model.per_layer_tok_embd_b, n_heads);
        }
        emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    }
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // column k of the [kern, hc_dim] kernel is one weight per channel
    std::vector<ggml_tensor *> w_taps;
    for (int64_t k = 0; k < kern; ++k) {
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }
        w_taps.push_back(wk);
    }

    // strixllama: a ragged ubatch (llama_ubatch::ragged) runs the conv once per group of sequence sets with one token
    // count, each on its own history; an equal-length ubatch is one group
    const bool ragged = ubatch.ragged();
    std::vector<ggml_tensor *> conv_parts;
    for (const auto & g : qwen4exp_seq_groups(ubatch)) {
        const int64_t n_grp_tok  = g.n_tok;
        const int64_t n_grp_seqs = g.n_seqs;

        ggml_tensor * x_g = ragged
            ? ggml_view_3d(ctx0, normalized, hc_dim, n_grp_tok, n_grp_seqs,
                    normalized->nb[1], normalized->nb[1] * n_grp_tok, normalized->nb[1] * g.t0)
            : ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs);

        // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
        ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il), x_g,
                hist, hc_dim, il, g.s0, ragged ? n_grp_seqs : -1);

        ggml_tensor * conv_g = nullptr;
        for (int64_t k = 0; k < kern; ++k) {
            // tap k reads (kern-1-k)*dilation positions back
            const int64_t start = hist - (kern - 1 - k) * dil;

            ggml_tensor * shifted = ggml_cont(ctx0,
                    ggml_transpose(ctx0,
                            ggml_view_3d(ctx0, padded, n_grp_tok, hc_dim, n_grp_seqs,
                                    padded->nb[1], padded->nb[2],
                                    ggml_row_size(padded->type, start))));

            ggml_tensor * term = ggml_mul(ctx0, shifted, w_taps[k]);
            conv_g = conv_g ? ggml_add(ctx0, conv_g, term) : term;
        }
        conv_parts.push_back(conv_g);
    }

    ggml_tensor * conv_out = ragged ? qwen4exp_concat_groups(ctx0, conv_parts, hc_dim) : conv_parts[0];

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
