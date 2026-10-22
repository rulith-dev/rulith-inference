# The recipe up to 0.6.4: pwilkin/llama.cpp @ f5daaa3cfa63, the whole files of patches/history/iq3s-kernel/,
# then these scripts in this order (all of them in patches/history/). Kept for the reasons each one gives;
# bootstrap no longer runs them - see patches/MANIFEST.md.

# Whole files, laid down before the scripts run. These are the IQ3_S MMB kernel and the sigmoid
# fusion: too large to express as anchored edits, and the scripts that follow edit ggml-cuda.cu on
# top of this version of it. llama-lazy-reader.h is deliberately NOT here - apply_win_lazy_reader
# writes it, and handing a patch its own finished output makes it skip the rest of its work.
SNAPSHOT = {"ggml-cuda.cu": "ggml/src/ggml-cuda/ggml-cuda.cu",
            "mmb.cu":       "ggml/src/ggml-cuda/mmb.cu",
            "mmb.cuh":      "ggml/src/ggml-cuda/mmb.cuh",
            "softcap.cu":   "ggml/src/ggml-cuda/softcap.cu",
            "softcap.cuh":  "ggml/src/ggml-cuda/softcap.cuh"}

# Order matters where several scripts write the same file; see patches/MANIFEST.md.
PATCH_ORDER = [
    "apply_win_lazy_reader",        # first: later patches anchor against the reader it installs
    "apply_mmvq_rdna35_rows", "apply_chain_fusion", "apply_getrows_cast_fusion",
    "apply_graph_key_shape", "apply_skip_ops_ablation", "apply_ple_overlapped_reads",
    "apply_qsa_block_key_cache", "apply_qsa_decode_gather", "apply_qsa_small_batch_mask",
    "apply_qsa_kb_image_guard", "apply_rs_pos_warn_stateless", "apply_spec_draft_ubatch",
    "apply_spec_timing", "apply_decode_timing", "apply_iq3s_vecdot", "apply_node_timing",
    "apply_mmid_sort_note", "apply_ple_prefetch_launch", "apply_trunk_twins",
    "apply_iq3s_mmq_hip",           # after apply_iq3s_vecdot: it rewrites the sign table that one added
    "apply_prompt_cache_disk", "apply_multi_stream_qsa", "apply_small_m_mmvf",      # after apply_spec_timing and friends: anchors in the patched server loop
    "apply_cache_ram_and_mtp",      # after apply_prompt_cache_disk: it rewrites the tier that one adds
    "apply_spc_direct_io",          # after it: the reader it replaces is the one that patch leaves
    "apply_disk_tier_v2",           # after both: it rewrites the disk tier they built
    "apply_hc_q8_fusions", "apply_qsa3_bitonic",
    "apply_disk_restore_lazy",      # after apply_disk_tier_v2: the restore path it changes is that one's
    "apply_ple_pregather",          # after apply_disk_restore_lazy: both edit the server loop
    "apply_state_copy_trim",        # after apply_ple_pregather: both edit qwen4exp.cpp
    "apply_moe_glu3",
    "apply_qsa_active_blocks", "apply_state_read_coalesce", "apply_kv_regions", "apply_compute_buffer_headroom", "apply_alloc_failure_report", "apply_empty_slot_cache",
    "apply_qsa_kb_rebuild_f16",
    "apply_disk_tier_v3",           # after apply_empty_slot_cache and the v2 tier: it adds to the server loop they patch
    "apply_slot_state_guard", "apply_disk_ckpt_step", "apply_disk_v3_vision",
    "apply_kv_q8_0", "apply_state_hash_debug",
    "apply_spec_draft_by_slots", "apply_kv_zero_freed", "apply_mtp_carry_state",
    "apply_gdn_direct_rows", "apply_ple_unbuffered_cache", "apply_prefill_kernels_020",
    "apply_spec_even_drafts", "apply_small_k_membership",
    "apply_bf16_mid_batches", "apply_q6k_mmq_rdna35",
    "apply_small_batch_decode",
    "apply_verify_small_products",
    "apply_image_dense_bound",
    "apply_prefix_cache_026", "apply_gdn_deferred_rollback", "apply_image_rank_per_seq",
    "apply_agent_slots_027", "apply_ragged_ubatch_027",
    "apply_qsa3_wide_window_028", "apply_prompt_alone_028", "apply_disk_store_v4_028",
    "apply_qsa_between_029", "apply_gdn_r16_lds_029", "apply_norm_rows_plain_029",
    "apply_kq_mask_seq_bits_030",
    "apply_kb_pending_031", "apply_mtp_index_share_031",
    "apply_ckpt_edges_032",
    "apply_pool_guard_033",
    "apply_idx_score_dec_034", "apply_qsa_runs_034", "apply_ckpt_budget_034",
    "apply_idx_score_skip_035", "apply_qsa_run_buffers_035", "apply_parallel_sampling_035", "apply_skip_ops_shape_035",
    "apply_restore_wait_035",
    "apply_mmvq_i8_036", "apply_idx_score_v3_036", "apply_kb_rows_036", "apply_top_k_rows_036", "apply_repeat_probe_036",
    "apply_fa_gather_037", "apply_mmvq_i8_unroll_037",
    "apply_bf16_twins_040", "apply_gpu_timeline_040",   # the second patches ggml-cuda.cu as the first leaves it
    "apply_decode_small_kernels_040", "apply_ab_switch_041", "apply_gdn_records_041", "apply_ple_wake_041",
    "apply_spec_sampling_042", "apply_mmb_kq_042",
    "apply_prefill_043",
    "apply_moe_glu_kq_perf",   # last: it anchors in the norm prints apply_prefill_043 adds
    "apply_xres_fix_044",       # after every patch that touches ggml-cuda.cu (0.4.4, issue #7)
    "apply_mtp_lowrank_head",   # 0.4.6: the draft head scored through a low-rank pre-score
    "apply_ot_host_buft",       # 0.4.6: -ot accepts ROCm_Host (weights outside the carve)
    "apply_host_buft_padding",  # 0.4.7: host buffers pad quantized rows (experts in system memory)
    "apply_dense_ubatch_own_cells",  # 0.4.7: issue #10, the image ubatch bound by the batch's own cells
    "apply_split_no_free_048",  # 0.4.8: an even layer split when every device reports no free memory
    "apply_ckpt_turn_cuts_048",  # 0.4.8: prompts cut only at the last user message
    "apply_poll_sync_049",      # 0.4.9: the stream wait polls host memory (ROCm/TheRock#8786)
    "apply_prefill_051",        # 0.5.1: delta-net records replayed on the GPU, fusions, expert kernels, MTP in place
    "apply_request_log_052",    # 0.5.2: the server's request log (STRIX_REQUEST_LOG)
    "apply_tool_schemas_052",   # 0.5.2: tool parameters as oneOf alternatives, allOf over non-object types
    "apply_tool_schemas_053",   # 0.5.3: tool parameters in any JSON Schema form; allOf requires what its parts require
    "apply_agent_compat_053",   # 0.5.3: Codex / Claude Code on /v1/responses and /v1/messages (issue #12)
    "apply_image_text_compact_054",   # 0.5.4: text after an image keeps the compact sparse-attention inputs (issue #13)
    "apply_own_tokens_055",   # 0.5.5: a conversation's next request keeps the tokens the model generated
    "apply_kb_uncached_f16_055",   # 0.5.5: uncached block keys rounded to F16 as the cache holds them (issue #16)
    "apply_server_056",   # 0.5.6: images in the splice, draft length by acceptance (#18), live draft length, MTP-off entries kept
    "apply_chat_056",   # 0.5.6: no thinking when asked for none (#17), one leading system message (#12)
    "apply_tool_params_056",   # 0.5.6: alternatives that each require a parameter: at least one in the grammar
    "apply_brand_057",         # 0.5.7: README, LICENSE, build info and the server start name Rulith Inference
    "apply_spec_adapt_057",    # 0.5.7: the draft length for several generating conversations measured on the server
    "apply_gdn_conv_norm_057", # 0.5.7: conv + q/k L2 norms in one pass; the conv's inputs kept from being overwritten
    "apply_ubatch_embd_057",   # 0.5.7: a ubatch of consecutive tokens reads the batch's embedding rows in place
    "apply_hc_cn_whole_058",   # 0.5.8: the HC combine + norm without per-iteration guards on 2560-wide streams
    "apply_mmvq_rows4_058",    # 0.5.8: four rows a workgroup for the single-column IQ4_NL product on RDNA3.5
    "apply_gdn_conv_decode_058",  # 0.5.8: the fused GDN conv + q/k norms for decode batches, state copies in the kernel
    "apply_topk_moe_spec_058", # 0.5.8: the softmax + top-k + norm router kernel as a compile-time instance
    "apply_dev_measure_058",   # 0.5.8: measurement only - STRIX_AB_MIN_NODES, STRIX_GPU_TIMELINE_PRINT
    "apply_nemotron_h_059",    # 0.5.9: Nemotron 3.5 Super loads (RMS eps fallback); its MTP block's latent MoE
    "apply_nemotron_omni_059", # 0.5.9: the nemotron_omni image projector (+ patches/nemotron-omni/nemotron-omni.cpp)
    "apply_mamba2_scan_059",   # 0.5.9: the Mamba-2 prompt scan from a state row per thread pair; new states into the cache
    "apply_concat_gaps_059",   # 0.5.9: CONCAT dim 0 from transposed views with gaps
    "apply_mmvq_q4k_i8_059",   # 0.5.9: Q4_K products of 2-8 columns through a lane-ordered y
    "apply_mtp_pos_jumps_059", # 0.5.9: an MTP draft context lets positions jump forward over a prompt's images
    "apply_exl3_ggml_060",     # 0.6.0: EXL3 weight types, MUL_MAT_HAD and QSA_PACK in ggml (+ patches/exl3/ggml-exl3.c/.h)
    "apply_exl3_cpu_060",      # 0.6.0: the CPU references of MUL_MAT_HAD and QSA_PACK (+ patches/exl3/mul-mat-had.cpp)
    "apply_exl3_hip_060",      # 0.6.0: EXL3 GEMV / GEMMs and fusions on the GPU, QSA_PACK (+ patches/exl3/mul-mat-had.cu/.cuh)
    "apply_exl3_llama_060",    # 0.6.0: EXL3 weights for qwen4exp; compute buffers that do not grow with the pool (+ llama-weight-aux)
    "apply_slot_continuation_060",  # 0.6.0: a conversation's next step continues in its own slot (issue #19)
    "apply_exl3_decode_063",   # 0.6.3: EXL3 dense products of up to 8 tokens through the GEMV (MTP verify)
    "apply_prompt_sample_063", # 0.6.3: EXL3 z + gated norm epilogue, one-pass dense prologue; exact fast top-k sampling
]
