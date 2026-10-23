#pragma once
// strixllama: the routed EXL3 experts of a prompt-sized batch as one pipeline (mmh-moe.cu): gate, up, SwiGLU, down and
// the weighted sum over the token's experts (plus the shared expert's gated tail) in 9 launches, with the rows sorted
// by expert in between. Plain types only, so ggml-cuda.cu and mmh-moe.cu share no headers.
#include <cstddef>
#include <cstdint>

struct ggml_cuda_mmh_moe_args {
    const float *   x;             // [T][H] rows, row stride x_ld floats (the experts' shared input)
    int64_t         x_ld;
    const int32_t * ids;           // [T] rows of n_used expert ids, row stride ids_ld
    int64_t         ids_ld;
    const float *   weights;       // [T][n_used] routing weights
    int             T, n_used, E, H, I;
    int             bits_gu, bits_d;
    const void *    w_gate, * w_up, * w_down;     // EXL3 weights, expert e at w + e * es (bytes)
    size_t          es_gate, es_up, es_down;
    const void *    rin_g, * rout_g, * rin_u, * rout_u, * rin_d, * rout_d;   // F16 rotations, expert e at + e * rs (halves)
    int64_t         rs_in_g, rs_out_g, rs_in_u, rs_out_u, rs_in_d, rs_out_d; // 0: one row shared by all experts
    const float *   sx;            // the shared expert's output [T][H] and its gate [T], or null
    const float *   sg;
    float *         out;           // [T][H]
    void *          buf_gu;        // 2 * A * I halves (A = T * n_used): the gate / up products, then the down input
    void *          buf_had;       // 2 * A * H halves: the rotated inputs of gate and up, then the down product (A * H floats)
    // kept from the main part to the reduce (ggml_cuda_mmh_moe_scratch::persist_bytes)
    int64_t *       inverse;       // [A] sorted position of each (token, pick)
    int32_t *       pick_e;        // [A] expert of each (token, pick)
    int64_t *       tables;        // [9][E] pointers: gate / up / down trellis, suh, svh
    // the main part's scratch (temp_bytes)
    int32_t *       sorted_e;      // [A] expert of each sorted position
    int32_t *       block_counts;  // [nb][E]
    int64_t *       counts;        // [E + 1]
    int64_t *       offsets;       // [E + 1]
    int32_t *       tiles;         // [slots]
    int32_t *       num_tiles;     // [1]
};

struct ggml_cuda_mmh_moe_scratch {
    int64_t A, nb, slots;
    size_t  persist_bytes;   // inverse, pick_e, tables: read again by the reduce
    size_t  temp_bytes;      // the rest, dead after the main part
};

ggml_cuda_mmh_moe_scratch ggml_cuda_mmh_moe_scratch_size(int T, int n_used, int E);
// carve the scratch arrays of `a` out of `persist` and `temp` (the sizes of ggml_cuda_mmh_moe_scratch_size)
void ggml_cuda_mmh_moe_scratch_carve(ggml_cuda_mmh_moe_args & a, void * persist, void * temp);
// whether the pipeline has kernels for these shapes and widths
bool ggml_cuda_mmh_moe_supported(int H, int I, int n_used, int bits_gu, int bits_d);
// sort, gather, gate / up, SwiGLU, down: buf_had then holds the down product's sorted F32 rows
void ggml_cuda_mmh_moe_run_main(const ggml_cuda_mmh_moe_args & a, void * stream);
// the weighted sum of the picks' rows (and the shared expert's gated tail) into out
void ggml_cuda_mmh_moe_run_reduce(const ggml_cuda_mmh_moe_args & a, void * stream);
