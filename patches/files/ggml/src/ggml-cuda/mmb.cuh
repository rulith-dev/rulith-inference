#pragma once
#include "common.cuh"
// MMB: dequant-to-BF16 WMMA GEMM path for IQ4_NL weights on gfx1151 (RDNA3.5). Env-gated: LLAMA_MMB=1, LLAMA_MMB_MIN_T (default 512).
bool ggml_cuda_mmb_supported_mm  (const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, bool allow_iq3s = false);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// strixllama: a narrow F32 weight (M <= 64) against any number of columns on WMMA (mmb_f32narrow_kernel); false when the
// shapes do not fit
bool ggml_cuda_mmb_f32_narrow(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// strixllama: F16 copies of narrow F32 weights for the two-product mode (made before a graph runs, outside any capture),
// and two narrow F32 weights against the same activations in one pass; false (nothing run) when it does not apply. Both
// leave every output bitwise what ggml_cuda_mmb_f32_narrow computes
void ggml_cuda_mmb_f16w_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
// strixllama: the GEMM node gemm (z = w x) with the gated RMS norm out = norm(o) * g * sigmoid(z) that is its only reader
// in one kernel, bitwise the two; false (nothing run) when it does not apply
bool ggml_cuda_mmb_gemm_gnorm(ggml_backend_cuda_context & ctx, const ggml_tensor * gemm, const ggml_tensor * o, const ggml_tensor * g,
        float eps, ggml_tensor * out);
bool ggml_cuda_mmb_f32_narrow_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * w1, const ggml_tensor * w2, const ggml_tensor * x,
        ggml_tensor * d1, ggml_tensor * d2);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph();
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// strixllama: the same for a [K, rows] output whose rows may be padded; *ld gets the row stride in elements (0 = K)
uint16_t * ggml_cuda_mmb_cache_reserve_ld(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t rows, int64_t K, int64_t * ld);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
// a caller passing ld can take padded rows (their stride in *ld, 0 = dense); one that does not only ever sees dense rows
const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t, int64_t * ld = nullptr);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n, int64_t ld = 0, size_t alloc_n = 0);
void ggml_cuda_mmb_marks_clear();
size_t ggml_cuda_mmb_marks_count();
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_gatemix();
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
// strixllama: whether ggml_cuda_hc_gate_mix will run for these tensors (the streams' copies aside), and whether it reads
// the F32 streams (*f32) - which it can instead form from a combine's residual output (ggml_cuda_hc_xres_set)
bool ggml_cuda_hc_gate_mix_ok(const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, const ggml_tensor * dst, int hc, bool * f32);
void ggml_cuda_hc_xres_set(const void * xn_data, const float * res, const float * row_scale, const float * gamma);
bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_release_all();
