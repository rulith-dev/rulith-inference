#pragma once

#include "llama-memory-hybrid.h"

#include "ggml-cpp.h"

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // strixllama: a sequence's attention rows by position, for the server's disk tier (llama_strix_kv_*): the
    // attention cache's rows, then the indexer's. The recurrent state is not in them - it is one per sequence,
    // and goes through a LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY state (a context checkpoint)
    size_t kv_row_size() const;
    bool   kv_rows_get(llama_seq_id seq_id, llama_pos p0, uint32_t n, uint8_t * dst) const;
    bool   kv_rows_set(llama_seq_id seq_id, llama_pos p0, uint32_t n, const uint8_t * src, uint32_t src_rows);
    // strixllama: the rows of seq_src at positions [p0, p0 + n) into seq_dst's cells for them (kv_alloc), on the device
    bool   kv_rows_copy(llama_seq_id seq_src, llama_seq_id seq_dst, llama_pos p0, uint32_t n);
    // seq_id loses everything it held and gets attention and indexer cells for positions [0, n) of `tokens`
    bool   kv_alloc(llama_seq_id seq_id, const llama_token * tokens, uint32_t n);

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
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
    // strixllama: a compact ubatch that serves several sequences (unified cache, equal-length split).
    // The position half of the visibility stays in `bias`; which sequence owns a block is a 0/1
    // membership matrix the scorer multiplies with the token's one-hot sequence slot (block-graph.inc):
    //   seq_blk F32 [n_seqs_unq, n_blocks]  1 where the block's cells belong to that sequence
    //   seq_tok F32 [n_seqs_unq, n_tokens]  one-hot slot of each token's sequence (ubatch.seq_idx)
    // Slots are the ubatch's own numbering (seq_id_unq), so the graph shape depends only on their count.
    struct qsa_mixed_inputs {
        ggml_tensor * seq_blk = nullptr;
        ggml_tensor * seq_tok = nullptr;
    };
    // strixllama: kv_off = the first cell of the graph's view of the caches (llama_kv_cache::get_kv_window).
    // Every cell index these inputs carry is relative to it, except the block-key cache rows, which address
    // the whole cache tensor.
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, const qsa_kb_inputs * kb = nullptr, uint32_t kv_off = 0) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio, const qsa_kb_inputs * kb = nullptr,
                             const qsa_mixed_inputs * mixed = nullptr, bool active_only = false, uint32_t kv_off = 0) const;

    ggml_tensor * get_kb(int32_t il) const;   // F16 [idx_dim, kb_r*kb_s]; null when off or no indexer on il
    uint32_t      kb_scratch_row() const;     // the spare row: kb_row(kv_size) of the indexer cache
    // strixllama: a block's key sits in the row of its first cell, and the rows are laid out by residue class: cell c
    // in row (c % kb_r)*kb_s + c/kb_r. A conversation's blocks start every kb_r cells (kb_r = the block ratio), so
    // their keys are adjacent rows, and the decode score reads them as one run. With a row per cell they were 1 KB
    // apart, and eight conversations' keys read that way ran at ~50 GB/s (205 us a layer at 8 x 20K, 45 us adjacent;
    // one conversation at 110K: 33 us, 17 us adjacent): the DRAM channels alias on strides like that. Any c -> row
    // bijection is correct whatever the block layout; this one is also dense.
    uint32_t      kb_row(uint32_t cell) const { return (cell % kb_r)*kb_s + cell/kb_r; }
    // strixllama: a sequence's block keys go stale when its positions move or its cells are restored or moved out
    // of order; the next graph of a ubatch holding it rebuilds every key of that ubatch's sequences
    bool          kb_needs_full(const llama_ubatch & ubatch) const;
    void          kb_mark_full(const llama_ubatch & ubatch) const;
    // strixllama: a graph that writes indexer keys without running the indexer (the dense shortcut, build_qsa_store_k)
    // refreshes no block key, so it leaves its sequences' blocks from its first position on pending (kb_from). The
    // next graph of a sequence that runs the indexer refreshes them: up to kb_pending_max positions besides its own
    // tokens in its dirty list, more with a full rebuild. Without this, a conversation's first ~2K tokens (a view that
    // small takes the shortcut) kept the block keys of the conversation the slot held before.
    static constexpr int32_t kb_pending_max = 64;
    void          kb_note_unrefreshed(const llama_ubatch & ubatch) const;
    // strixllama: true once a ubatch with per-axis positions (an image under M-RoPE) has been written.
    // Such cells repeat one position across the image, so set_input_qsa ranks cells instead of using
    // the position, which the block-key cache cannot track - the graph must not wire the cache in.
    bool          kb_pos_dup() const;

    // strixllama: IndexShare for a QSA MTP draft context (LLAMA_MTP_INDEX_SHARE=1, llama_set_mtp_share_reuse). A catch-up
    // that runs the indexer keeps the sparse-attention selection of the positions it decodes in a ring of share_ring rows
    // per sequence (row seq*share_ring + pos % share_ring). A draft step then attends to the latest selection kept before
    // it plus every cell of the positions since, instead of running the indexer; a catch-up runs it again only when that
    // selection is more than a refresh distance behind. A row holds absolute cell + 1 for each selected cell (0 = none), so
    // it survives the graph's view of the cache moving; whatever drops, moves or rewrites a sequence's cells forgets its rows.
    //   share_sel F32 [share_width, n_seq_max*share_ring + 2]: the rows, then a scratch row and an all-zero row
    // a draft step's extra columns (cells of the positions after the reused selection) and the widest gap it may cover:
    // the rest is room for 16 draft positions
    static constexpr int32_t share_extra   = 64;
    static constexpr int32_t share_max_gap = share_extra - 16;
    ggml_tensor * share_sel()   const { return share_t; }
    int64_t       share_width() const { return share_w; }
    llama_seq_id  share_n_seqs() const { return (llama_seq_id) share_n_seq; }
    uint32_t      share_ring_len() const { return share_ring; }
    int32_t       share_row(llama_seq_id seq_id, llama_pos pos) const;
    int32_t       share_scratch_row() const;
    int32_t       share_none_row() const;
    void          share_note(llama_seq_id seq_id, llama_pos pos) const;   // a graph being set up captures (seq_id, pos)
    bool          share_ready(llama_seq_id seq_id, llama_pos pos) const;  // a selection is kept for exactly pos
    llama_pos     share_latest(llama_seq_id seq_id, llama_pos below) const;   // the latest kept position < below, -1: none
    // the absolute cells of seq_id's positions in [p0, p1], in position order
    void          share_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<int64_t> & out) const;
    // whether a catch-up ubatch has to run the indexer to keep selections: a sequence of it has none kept before it, or
    // the latest is more than `refresh` positions behind the ubatch's last
    bool          share_capture_due(const llama_ubatch & ubatch, llama_pos refresh) const;

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    void set_input_qsa_impl(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs,
                            const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias,
                            const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed = nullptr,
                            bool active_only = false, uint32_t kv_off = 0) const;
    // strixllama: what set_input_qsa_scan writes, for one conversation that is one run (llama_kv_cells::seq_run)
    struct qsa_run_inputs {
        std::vector<int32_t> blk_cells, blk_pos, limits, tail, bid_rows, dirty_cells, dirty_pos, dirty_dst;
        std::vector<float>   seq_blk, seq_tok;
        std::vector<llama_seq_id> clear_from;   // kb_from[s] = none for these
        bool set_stale = false;                 // and kb_stale[s] = 1
    };
    // kept from one call to the next, so a step's inputs reuse the last one's memory (eight conversations at 40K: ~4 MB)
    mutable qsa_run_inputs       run_ri;
    mutable std::vector<int32_t> run_blk_run, run_blk_pb;
    bool set_input_qsa_run(qsa_run_inputs & out, ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                           ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio,
                           const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed, bool active_only, uint32_t kv_off) const;
    void set_input_qsa_scan(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs,
                            const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias,
                            const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed,
                            bool active_only, uint32_t kv_off) const;

    // strixllama: block-key cache storage, one tensor per indexer layer, in the layer's device buffer
    std::vector<ggml_context_ptr>        kb_ctxs;
    std::vector<ggml_backend_buffer_ptr> kb_bufs;
    std::map<int32_t, ggml_tensor *>     kb_map;
    uint32_t                             kb_r = 1;   // rows laid out by residue class mod kb_r (kb_row)
    uint32_t                             kb_s = 0;   // rows a class: ceil((kv_size + 1)/kb_r)
    // the sequences whose block keys have to be rebuilt (see kb_needs_full). All of them at the start and after
    // a clear, so the first graph of every sequence builds its keys the way it always has
    mutable std::bitset<LLAMA_MAX_SEQ>   kb_stale = std::bitset<LLAMA_MAX_SEQ>().set();
    void kb_mark_stale(llama_seq_id seq_id);   // < 0: every sequence
    // per sequence, the first position whose block key a graph left pending (kb_note_unrefreshed), max() for none
    mutable std::vector<llama_pos>       kb_from = std::vector<llama_pos>(LLAMA_MAX_SEQ, std::numeric_limits<llama_pos>::max());
    // the positions of seq_id's pending span outside [pmin, pmax], the ubatch's own positions of it (0: none pending)
    llama_pos kb_pending_extra(llama_seq_id seq_id, llama_pos pmin, llama_pos pmax) const;
    // strixllama: regions - block-key rows follow their cells (llama_kv_cache::move_cells)
    void kb_move_rows(const llama_kv_cache::cell_move_vec_t & moves);
    // set in init_batch when an image ubatch or a position gap arrives, cleared only when every
    // sequence is dropped - see kb_pos_dup() and the reset in seq_rm()
    bool                                 kb_dup      = false;

    // strixllama: IndexShare storage and bookkeeping (see share_sel)
    ggml_context_ptr                     share_ctx;
    ggml_backend_buffer_ptr              share_buf;
    ggml_tensor *                        share_t     = nullptr;
    int64_t                              share_w     = 0;
    uint32_t                             share_ring  = 8;
    uint32_t                             share_n_seq = 0;
    mutable std::vector<llama_pos>       share_pos;     // the position each ring row holds, -1 none
    // drop seq_id's rows for positions in [p0, p1) (p1 < 0: to the end); seq_id < 0: every sequence
    void share_forget(llama_seq_id seq_id, llama_pos p0 = -1, llama_pos p1 = -1) const;

    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;
    bool qsa_scalar_visibility(const llama_ubatch & ubatch) const;
    bool qsa_position_prefix(const llama_ubatch & ubatch, bool active_only = false) const;
    // strixllama: the length of a block list that covers only the ubatch's own sequences, at least min_blocks
    // (the selection's budget), or 0 when the list must cover every cell of the pool (see the definition)
    int64_t qsa_active_blocks(const llama_ubatch & ubatch, uint32_t ratio, int64_t min_blocks) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, const llama_memory_hybrid_idx::qsa_kb_inputs * kb = nullptr) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                             ggml_tensor * bias, ggml_tensor * tail_idxs,
                             const llama_ubatch * ubatch, uint32_t ratio,
                             const llama_memory_hybrid_idx::qsa_kb_inputs * kb = nullptr,
                             const llama_memory_hybrid_idx::qsa_mixed_inputs * mixed = nullptr,
                             bool active_only = false) const;

    // strixllama: block-key cache pass-throughs (see llama_memory_hybrid_idx)
    ggml_tensor * get_kb(int32_t il) const;
    uint32_t      kb_scratch_row() const;
    bool          kb_needs_full(const llama_ubatch & ubatch) const;
    void          kb_mark_full(const llama_ubatch & ubatch) const;
    void          kb_note_unrefreshed(const llama_ubatch & ubatch) const;
    bool          kb_pos_dup() const;

    // strixllama: IndexShare pass-throughs (see llama_memory_hybrid_idx::share_sel)
    const llama_memory_hybrid_idx * get_mem() const { return mem; }

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
