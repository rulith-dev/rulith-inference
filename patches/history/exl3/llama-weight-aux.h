#pragma once

// strixllama: tensors that a weight needs beside its own data - EXL3 weights (ExLlamaV3, MIT, turboderp) are used with
// ggml_mul_mat_had and need <base>.rot_in and <base>.rot_out. From ranma.cpp (saturnsky), MIT license, without its
// engram table pieces and dimension padding (qwen4exp's dims are multiples of 128).

#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>

struct llama_weight_aux {
    ggml_tensor * rot_in  = nullptr; // [ne0(, ne2)]
    ggml_tensor * rot_out = nullptr; // [ne1(, ne2)]
    // strixllama: per-expert widths: the weight is class 0 of the experts, wc[0] / wc[1] classes 1 / 2 (<base>.c1/.c2
    // .weight), cmap I32 [E] (<base>.cmap: class << 24 | index in the class); the rotations are then [.., E]
    ggml_tensor * cmap  = nullptr;
    ggml_tensor * wc[2] = { nullptr, nullptr };
};

// key: the weight tensor of the model
using llama_weight_aux_map = std::unordered_map<const ggml_tensor *, llama_weight_aux>;

// w * cur (ggml_mul_mat), or with ids the expert product (ggml_mul_mat_id), for a weight that needs aux tensors
// w can also be a view of whole 128-row groups of such a weight
// returns nullptr if w needs no aux tensors; aborts for an EXL3 weight that is not in waux
ggml_tensor * llama_weight_aux_mul_mat(
        ggml_context               * ctx,
        const llama_weight_aux_map * waux,
        ggml_tensor                * w,
        ggml_tensor                * cur,
        ggml_tensor                * ids);

// the rows of an EXL3 row codec table (ggml_is_exl3_row) as ExLlamaV3's ngram_dequant gives them: fp16(row + bias[head])
// with the head the fastest index of rows (n_heads per token); rows = what ggml_get_rows(table, ...) returned, F32
// [ne0, n_heads * n_tokens]; bias F16/F32 [ne0, n_heads]
ggml_tensor * llama_weight_aux_row_bias(
        ggml_context * ctx,
        ggml_tensor  * rows,
        ggml_tensor  * bias,
        int64_t        n_heads);

// a graph node that uses w as the model would (the device support check of the loader): MUL_MAT_HAD with rotations of
// the right shape, n_ids > 0 for experts
ggml_tensor * llama_weight_aux_test_op(ggml_context * ctx, ggml_tensor * w, int n_ids);

// "<base>" of an EXL3 weight "<base>.weight"; throws for any other name
std::string llama_weight_aux_base(const std::string & name);
