#!/usr/bin/env python3
r"""Cache the QSA indexer block keys instead of rebuilding them from the whole cache every graph.

Per-node timing of a decode graph at 97K context (results/hip-rocm101-20260917.json,
decode_node_timing_97k_20260918) showed that half of the context-proportional cost was not attention
but the indexer key path: every graph gathered ALL cached raw indexer keys (GET_ROWS over 97536 rows,
2.6 ms), mean-pooled them into 24384 blocks, rms-normed (2.0 ms) and roped (4.3 ms) every block, per
decoded token. A finished block's key depends only on its member cells and its position, so it can be
built once, when a write completes the block, and read back afterwards.

  llama-memory-hybrid-idx.{h,cpp}
    - one F16 [idx_dim, kv_size + 1] tensor per indexer layer in the layer's device buffer (12 x 64 MiB at
      ctx 262144), a block's key at the row of its first member cell, the last row a scratch target
    - set_input_qsa_impl gains four outputs: the member cells, positions and destination rows of the
      blocks this ubatch completes (padded to n_tokens/ratio + 4), and the row of every enumerated block
    - a generation counter invalidates everything on seq_add / seq_div / state_read / clear; the next
      graph rebuilds all keys and rewrites the cache (marked when that graph's inputs are set, because
      graph_reserve builds graphs that never run)
  models/qwen4exp.cpp
    - the incremental graph builds only the dirty blocks, scatters them with ggml_set_rows and gathers
      every enumerated block's key with ggml_get_rows; the full-rebuild graph keeps the old path and
      also writes the cache; can_reuse pins the mode and the dirty-list size
    - single stream and 1-D positions only (multi-stream or mrope images fall back to the old path)

LLAMA_QSA_BLOCK_KEY_CACHE=0 disables it.

Usage: python patches/apply_qsa_block_key_cache.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
H = os.path.join(tree, "src", "llama-memory-hybrid-idx.h")
CPP = os.path.join(tree, "src", "llama-memory-hybrid-idx.cpp")
MODEL = os.path.join(tree, "src", "models", "qwen4exp.cpp")

H_EDITS = [
    ('''#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>''', '''#include "llama-memory-hybrid.h"

#include "ggml-cpp.h"

#include <map>
#include <memory>
#include <vector>'''),
    ('''    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio) const;

private:
    void set_input_qsa_impl(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs,
                            const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;''',
     '''    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    //
    // strixllama: incremental block-key cache (LLAMA_QSA_BLOCK_KEY_CACHE, default on). The pooled, normed and
    // rotated indexer key of a block depends only on its member cells and its position, so it is computed
    // once, when a write completes the block, and kept in one F16 row per cell of the indexer cache (at the
    // row of the block's first member cell) plus one scratch row. Before it, every graph rebuilt every block
    // key from the raw cache: a full-context gather, pool, norm and rope per decoded token (10 ms at 97K).
    //   dirty_cells I32 [ratio*dirty_max]  member cells of the blocks this ubatch completes (padded)
    //   dirty_pos   I32 [4*dirty_max]      their mrope position rows, laid out like blk_pos
    //   dirty_dst   I32 [dirty_max]        destination rows (padding writes the scratch row)
    //   bid_rows    I32 [n_blocks]         row of every enumerated block, scratch row past n_bid
    struct qsa_kb_inputs {
        ggml_tensor * dirty_cells = nullptr;
        ggml_tensor * dirty_pos   = nullptr;
        ggml_tensor * dirty_dst   = nullptr;
        ggml_tensor * bid_rows    = nullptr;
    };
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, const qsa_kb_inputs * kb = nullptr) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio, const qsa_kb_inputs * kb = nullptr) const;

    ggml_tensor * get_kb(int32_t il) const;   // F16 [idx_dim, kv_size + 1]; null when off or no indexer on il
    uint32_t      kb_scratch_row() const;     // the spare row: kv_size of the indexer cache
    bool          kb_needs_full() const;      // positions moved (shift, restore, clear) since the last full write
    void          kb_mark_full() const;

private:
    void set_input_qsa_impl(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs,
                            const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias,
                            const qsa_kb_inputs * kb) const;

    // strixllama: block-key cache storage, one tensor per indexer layer, in the layer's device buffer
    std::vector<ggml_context_ptr>        kb_ctxs;
    std::vector<ggml_backend_buffer_ptr> kb_bufs;
    std::map<int32_t, ggml_tensor *>     kb_map;
    uint64_t                             kb_gen      = 1;
    mutable uint64_t                     kb_full_gen = 0;'''),
    ('''    bool qsa_position_prefix(const llama_ubatch & ubatch) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio) const;''',
     '''    bool qsa_position_prefix(const llama_ubatch & ubatch) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, const llama_memory_hybrid_idx::qsa_kb_inputs * kb = nullptr) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio,
                             const llama_memory_hybrid_idx::qsa_kb_inputs * kb = nullptr) const;

    // strixllama: block-key cache pass-throughs (see llama_memory_hybrid_idx)
    ggml_tensor * get_kb(int32_t il) const;
    uint32_t      kb_scratch_row() const;
    bool          kb_needs_full() const;
    void          kb_mark_full() const;'''),
]

CPP_SIGS = [
    'void llama_memory_hybrid_idx::clear(bool data) {',
    'void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {',
    'void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {',
    'void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {',
]
CPP_HOOK = '\n    kb_gen++;   // strixllama: block keys depend on positions and cell contents; rebuild them all once'

CPP_EDITS = [
    ('''            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}''', '''            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // strixllama: block-key cache, one F16 [idx_dim, kv_size + 1] tensor per indexer layer, allocated next to
    // that layer's indexer keys (LLAMA_QSA_BLOCK_KEY_CACHE=0 turns it off and restores the full rebuild)
    const char * kb_env = getenv("LLAMA_QSA_BLOCK_KEY_CACHE");
    if (mem_idx && (kb_env == nullptr || atoi(kb_env) != 0)) {
        const uint32_t kv_size = mem_idx->get_size();
        const int64_t  idx_dim = model.hparams.indexer_head_size;
        const uint32_t n_layer = model.hparams.n_layer_all;

        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!filter_idx(il)) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (offload) {
                buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
            }
            ggml_context * ctx = nullptr;
            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = { size_t(2u*n_layer*ggml_tensor_overhead()), nullptr, true };
                ctx = ggml_init(params);
                if (ctx == nullptr) {
                    throw std::runtime_error("failed to create ggml context for the QSA block-key cache");
                }
                ctx_map[buft] = ctx;
                kb_ctxs.emplace_back(ctx);
            } else {
                ctx = it->second;
            }
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, idx_dim, (int64_t) kv_size + 1);
            ggml_format_name(t, "cache_idx_kb_l%d", il);
            kb_map[(int32_t) il] = t;
        }
        size_t bytes = 0;
        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error("failed to allocate the QSA block-key cache");
            }
            ggml_backend_buffer_clear(buf, 0);
            bytes += ggml_backend_buffer_get_size(buf);
            kb_bufs.emplace_back(buf);
        }
        LLAMA_LOG_INFO("%s: QSA block-key cache: %zu layers, %.1f MiB\\n", __func__, kb_map.size(), bytes/1024.0/1024.0);
    }
}

ggml_tensor * llama_memory_hybrid_idx::get_kb(int32_t il) const {
    const auto it = kb_map.find(il);
    return it == kb_map.end() ? nullptr : it->second;
}

uint32_t llama_memory_hybrid_idx::kb_scratch_row() const {
    return mem_idx ? mem_idx->get_size() : 0;
}

bool llama_memory_hybrid_idx::kb_needs_full() const {
    return kb_full_gen != kb_gen;
}

void llama_memory_hybrid_idx::kb_mark_full() const {
    kb_full_gen = kb_gen;
}'''),
    ('''        ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias);
}''', '''        ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias, const qsa_kb_inputs * kb) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias, kb);
}'''),
    ('''        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true);
}''', '''        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio, const qsa_kb_inputs * kb) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true, kb);
}'''),
    ('''        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);''', '''        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const qsa_kb_inputs * kb) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);'''),
    ('''        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}''', '''        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, kb);
}

ggml_tensor * llama_memory_hybrid_idx_context::get_kb(int32_t il) const {
    return mem ? mem->get_kb(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::kb_scratch_row() const {
    return mem ? mem->kb_scratch_row() : 0;
}

bool llama_memory_hybrid_idx_context::kb_needs_full() const {
    return mem ? mem->kb_needs_full() : false;
}

void llama_memory_hybrid_idx_context::kb_mark_full() const {
    if (mem) { mem->kb_mark_full(); }
}'''),
    ('''        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio);
}''', '''        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kb);
}'''),
    ('''    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;''', '''    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    // strixllama: with the block-key cache an incremental graph builds its keys from the dirty list, so no
    // node reads blk_cells / blk_pos and ggml leaves them unallocated. They are still needed here (the
    // dirty-block scan reads them), so fall back to local scratch instead of writing through null.
    std::vector<int32_t> blk_cells_local, blk_pos_local;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    if (dst_blk_cells == nullptr) {
        blk_cells_local.assign((size_t) blk_cells->ne[0]*blk_cells->ne[1], 0);
        dst_blk_cells = blk_cells_local.data();
    }
    int32_t * dst_blk_pos = (int32_t *) blk_pos->data;
    if (dst_blk_pos == nullptr) {
        blk_pos_local.assign((size_t) blk_pos->ne[0], 0);
        dst_blk_pos = blk_pos_local.data();
    }'''),
    ('''                limits[b] = b<n_bid && cells.seq_has((uint32_t)bid_cell[b],seq) ? bid_idx[b] : INT32_MAX;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {''', '''                limits[b] = b<n_bid && cells.seq_has((uint32_t)bid_cell[b],seq) ? bid_idx[b] : INT32_MAX;
            }
        }

        // strixllama: block-key cache inputs. A cached block key goes stale only when one of the block's cells
        // is written, and a write happens exactly when a ubatch token lands in the cell, so the blocks this
        // ubatch completes are the full groups holding a cell whose (seq, pos) is a ubatch token.
        if (kb) {
            GGML_ASSERT(n_ns == 1 && !ranked && "qsa block-key cache: single stream, 1-D positions");
            GGML_ASSERT(kb->bid_rows && kb->bid_rows->ne[0] == n_blocks && kb->bid_rows->data);
            const int32_t scratch = (int32_t) kb_scratch_row();
            int32_t * br = (int32_t *) kb->bid_rows->data;
            for (int64_t b = 0; b < n_blocks; ++b) {
                br[b] = b < n_bid ? bid_cell[b] : scratch;
            }
        }
        // the dirty list exists only in incremental graphs (a full-rebuild graph reads no such input)
        if (kb && kb->dirty_dst && kb->dirty_dst->data) {
            const int64_t dirty_max = kb->dirty_dst->ne[0];
            GGML_ASSERT(kb->dirty_cells->ne[0] == r*dirty_max && kb->dirty_pos->ne[0] == 4*dirty_max);
            GGML_ASSERT(kb->dirty_cells->data && kb->dirty_pos->data);
            const int32_t scratch = (int32_t) kb_scratch_row();

            llama_pos pmin = ubatch->pos[0], pmax = ubatch->pos[0];
            for (int64_t i = 1; i < n_tokens; ++i) {
                pmin = std::min(pmin, ubatch->pos[i]);
                pmax = std::max(pmax, ubatch->pos[i]);
            }
            std::vector<uint8_t> grp_dirty(grp_first.size(), 0);
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];
                if (g < 0 || grp_bid[g] < 0) { continue; }
                const llama_pos p = cells.pos_get(j);
                if (p < pmin || p > pmax) { continue; }
                for (int64_t i = 0; i < n_tokens; ++i) {
                    if (ubatch->pos[i] == p && cells.seq_has(j, ubatch->seq_id[i][0])) { grp_dirty[g] = 1; break; }
                }
            }

            int32_t * dc = (int32_t *) kb->dirty_cells->data;
            int32_t * dp = (int32_t *) kb->dirty_pos->data;
            int32_t * dd = (int32_t *) kb->dirty_dst->data;
            int64_t n_dirty = 0;
            int64_t n_dirty_total = 0;
            for (int32_t b = 0; b < n_bid; ++b) {
                bool dirty = false;
                for (int64_t slot = 0; slot < r; ++slot) {
                    const int32_t g = cell_grp[cur_blk_cells[b*r + slot]];
                    if (g >= 0 && grp_dirty[g]) { dirty = true; break; }
                }
                if (!dirty) { continue; }
                ++n_dirty_total;
                if (n_dirty >= dirty_max) { continue; }
                for (int64_t slot = 0; slot < r; ++slot) { dc[n_dirty*r + slot] = cur_blk_cells[b*r + slot]; }
                for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + n_dirty] = dst_blk_pos[sec*n_blocks + b]; }
                dd[n_dirty] = bid_cell[b];
                ++n_dirty;
            }
            GGML_ASSERT(n_dirty_total <= dirty_max && "qsa block-key cache: more blocks completed than the graph can refresh");
            for (int64_t d = n_dirty; d < dirty_max; ++d) {
                for (int64_t slot = 0; slot < r; ++slot) { dc[d*r + slot] = (int32_t) std::min<int64_t>(slot, n_kv - 1); }
                for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + d] = 0; }
                dd[d] = scratch;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {'''),
]

MODEL_EDITS = [
    ('''static bool qwen4exp_qsa_flag(const char *name) {''', '''// the mrope sections make every ubatch "2-D" (n_pos > 1); text has all rows equal, images do not
static bool qwen4exp_pos_scalar(const llama_ubatch & ubatch) {
    if (!ubatch.pos || ubatch.n_tokens == 0) { return false; }
    for (uint32_t axis = 1; axis < ubatch.n_pos; ++axis) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) { return false; }
        }
    }
    return true;
}

static bool qwen4exp_qsa_flag(const char *name) {'''),
    ('''    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]''', '''    // strixllama: block-key cache inputs (null when the cache is off); kb_full = this graph rebuilds every key
    ggml_tensor * kb_dirty_cells = nullptr;   // I32 [ratio*dirty_max, 1]
    ggml_tensor * kb_dirty_pos   = nullptr;   // I32 [4*dirty_max]
    ggml_tensor * kb_dirty_dst   = nullptr;   // I32 [dirty_max]
    ggml_tensor * kb_bid_rows    = nullptr;   // I32 [n_blocks]
    bool          kb_full        = false;

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]'''),
    ('''    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        if (tail_idxs) {
            mctx->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio);
        } else {
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
        }
    }''', '''    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        llama_memory_hybrid_idx::qsa_kb_inputs kb;
        kb.dirty_cells = kb_dirty_cells; kb.dirty_pos = kb_dirty_pos; kb.dirty_dst = kb_dirty_dst; kb.bid_rows = kb_bid_rows;
        const auto * kbp = kb_bid_rows ? &kb : nullptr;
        if (tail_idxs) {
            mctx->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kbp);
        } else {
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, kbp);
        }
        // a graph that rewrites every block key refreshes the cache when it runs, not when it is built:
        // llama_context::graph_reserve builds graphs that never execute
        if (kbp && kb_full) {
            mctx->kb_mark_full();
        }
    }'''),
    ('''        res &= score_strip==next_strip;
        res &= score_key_limits==next_limits;

        return res;
    }''', '''        res &= score_strip==next_strip;
        res &= score_key_limits==next_limits;
        // strixllama: block-key cache shapes and mode are baked into the graph
        if (kb_bid_rows) {
            res &= kb_full == mctx->kb_needs_full();
            res &= kb_bid_rows->ne[0] == n_blocks;
            if (kb_dirty_dst) { res &= kb_dirty_dst->ne[0] == params.ubatch.n_tokens/n_stream/ratio + 4; }
        }

        return res;
    }'''),
    ('''            qsa->tail_idxs=ggml_new_tensor_3d(ctx0,GGML_TYPE_I32,r-1,n_tps,n_stream);
            ggml_set_input(qsa->tail_idxs);
        }''', '''            qsa->tail_idxs=ggml_new_tensor_3d(ctx0,GGML_TYPE_I32,r-1,n_tps,n_stream);
            ggml_set_input(qsa->tail_idxs);
        }
        // strixllama: block-key cache inputs (llama_memory_hybrid_idx::qsa_kb_inputs); single stream, 1-D positions only
        if (mctx_hyb->get_kb(il) != nullptr && n_stream == 1 && qwen4exp_pos_scalar(ubatch)) {
            const int64_t dirty_max = n_tps/r + 4;
            // an input no node reads is never allocated, so a full-rebuild graph gets only the rows tensor
            qsa->kb_full        = mctx_hyb->kb_needs_full();
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
        }'''),
]

MODEL_GRAPH_START = '    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].'
MODEL_GRAPH_END = '    cb(pooled, "indexer_k", il);\n'
MODEL_GRAPH_NEW = '''    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
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
        }
    }
    cb(pooled, "indexer_k", il);
'''


def apply(path, edits):
    s = io.open(path, encoding="utf-8").read()
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit("anchor not unique in %s: %r" % (path, old[:70]))
    for old, new in edits:
        s = s.replace(old, new)
    return s


def main():
    h = io.open(H, encoding="utf-8").read()
    if "qsa_kb_inputs" in h:
        print("already applied")
        return
    h = apply(H, H_EDITS)
    cpp = apply(CPP, CPP_EDITS)
    for sig in CPP_SIGS:
        if cpp.count(sig) != 1:
            sys.exit("hook anchor not unique: %s" % sig)
        cpp = cpp.replace(sig, sig + CPP_HOOK)
    model = apply(MODEL, MODEL_EDITS)
    a = model.index(MODEL_GRAPH_START)
    b = model.index(MODEL_GRAPH_END, a) + len(MODEL_GRAPH_END)
    model = model[:a] + MODEL_GRAPH_NEW + model[b:]
    for path, text in ((H, h), (CPP, cpp), (MODEL, model)):
        io.open(path, "w", encoding="utf-8", newline="").write(text)
        print("patched", path)


if __name__ == "__main__":
    main()
