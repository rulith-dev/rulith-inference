#include "llama-weight-aux.h"

#include "llama-impl.h"

#include <stdexcept>

// strixllama: from ranma.cpp (saturnsky), MIT license (see llama-weight-aux.h)

ggml_tensor * llama_weight_aux_mul_mat(
        ggml_context               * ctx,
        const llama_weight_aux_map * waux,
        ggml_tensor                * w,
        ggml_tensor                * cur,
        ggml_tensor                * ids) {
    if (!ggml_is_exl3(w->type)) {
        return nullptr;
    }

    const ggml_tensor * src = w->view_src ? w->view_src : w;
    const auto it = waux ? waux->find(src) : llama_weight_aux_map::const_iterator();
    if (waux == nullptr || it == waux->end()) {
        GGML_ABORT("EXL3 weight '%s' has no rot_in / rot_out, it must be a weight created by the model loader", w->name);
    }
    const llama_weight_aux & aux = it->second;

    ggml_tensor * rot_in  = aux.rot_in;
    ggml_tensor * rot_out = aux.rot_out;

    if (aux.cmap) {                    // strixllama: per-expert widths
        if (w->view_src || ids == nullptr) {
            GGML_ABORT("EXL3 weight '%s' has expert classes: only whole expert products with ids", w->name);
        }
        if (cur->nb[0] != ggml_element_size(cur)) {
            cur = ggml_cont(ctx, cur);
        }
        return ggml_mul_mat_had_classes(ctx, w, cur, rot_in, rot_out, ids, 128, aux.cmap, aux.wc[0], aux.wc[1]);
    }

    if (w->view_src) {
        // whole 128-row groups of the weight, for one expert or a range of experts
        const size_t  offs = w->view_offs;
        const int64_t e0   = offs / src->nb[2];
        const int64_t r0   = (offs % src->nb[2]) / src->nb[1];
        if (w->ne[0] != src->ne[0] || w->nb[1] != src->nb[1] || (w->ne[2] > 1 && w->nb[2] != src->nb[2]) ||
                (offs % src->nb[2]) % (128*src->nb[1]) != 0 || r0 + w->ne[1] > src->ne[1] || e0 + w->ne[2] > src->ne[2]) {
            GGML_ABORT("view '%s' of EXL3 weight '%s' does not take whole 128-row groups", w->name, src->name);
        }
        // a shared rotation ([n, 1]) stays shared for any range of experts
        rot_in  = rot_in->ne[1] == 1 ? rot_in :
            ggml_view_2d(ctx, rot_in,  rot_in->ne[0], w->ne[2], rot_in->nb[1],  e0*rot_in->nb[1]);
        rot_out = rot_out->ne[1] == 1 ? ggml_view_2d(ctx, rot_out, w->ne[1], 1, rot_out->nb[1], r0*rot_out->nb[0]) :
            ggml_view_2d(ctx, rot_out, w->ne[1],      w->ne[2], rot_out->nb[1], e0*rot_out->nb[1] + r0*rot_out->nb[0]);
    }

    if (cur->nb[0] != ggml_element_size(cur)) {
        cur = ggml_cont(ctx, cur);
    }

    return ggml_mul_mat_had(ctx, w, cur, rot_in, rot_out, ids, 128);
}

ggml_tensor * llama_weight_aux_row_bias(
        ggml_context * ctx,
        ggml_tensor  * rows,
        ggml_tensor  * bias,
        int64_t        n_heads) {
    const int64_t dim = rows->ne[0];
    GGML_ASSERT(bias != nullptr && bias->ne[0] == dim && bias->ne[1] == n_heads && ggml_nrows(bias) == n_heads);
    GGML_ASSERT(ggml_nrows(rows) % n_heads == 0);
    const int64_t n = ggml_nrows(rows) / n_heads;

    // ExLlamaV3 ngram_dequant: fp16(codebook * scale + bias[head]), the sum in f32
    // get_rows gives codebook * scale in f32 (exact), F16 -> F32 of the bias is exact, the add rounds once in f32
    ggml_tensor * res = ggml_reshape_3d(ctx, ggml_cont(ctx, rows), dim, n_heads, n);
    res = ggml_add(ctx, res, bias->type == GGML_TYPE_F32 ? bias : ggml_cast(ctx, bias, GGML_TYPE_F32));
    res = ggml_cast(ctx, ggml_cast(ctx, res, GGML_TYPE_F16), GGML_TYPE_F32);

    return ggml_reshape_2d(ctx, res, dim, n_heads * n);
}

ggml_tensor * llama_weight_aux_test_op(ggml_context * ctx, ggml_tensor * w, int n_ids) {
    ggml_tensor * rot_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, w->ne[0], w->ne[2]);
    ggml_tensor * rot_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, w->ne[1], w->ne[2]);
    if (n_ids > 0) {
        ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_ids, 512);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, 512);
        return ggml_mul_mat_had(ctx, w, b, rot_in, rot_out, ids, 128);
    }
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
    return ggml_mul_mat_had(ctx, w, b, rot_in, rot_out, nullptr, 128);
}

std::string llama_weight_aux_base(const std::string & name) {
    static const std::string suffix = ".weight";
    if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error(format("EXL3 tensor '%s' is not a .weight tensor", name.c_str()));
    }
    return name.substr(0, name.size() - suffix.size());
}
