#include "strixllama-chain.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STRIX_CHAIN_MAX 8

enum strixllama_chain_op : int32_t { FC_SCALE = 0, FC_CLAMP, FC_UNARY, FC_ADD, FC_SUB, FC_MUL, FC_DIV };

struct strixllama_chain_step {
    int32_t       op;            // strixllama_chain_op
    int32_t       unary;         // ggml_unary_op when op == FC_UNARY
    int32_t       chain_is_src1; // binary ops: the running value is the right-hand operand
    float         p0, p1;        // scale/bias or clamp min/max
    const char *  other;         // binary operand (or the source of a folded REPEAT)
    int64_t       one[4];        // its ne
    int64_t       onb[4];        // its nb (bytes)
};

struct strixllama_chain_params {
    const char * in;
    int64_t      in_ne[4];
    int64_t      in_nb[4];
    float *      out;
    int64_t      out_ne[4];
    int64_t      n;
    int32_t      n_steps;
    strixllama_chain_step steps[STRIX_CHAIN_MAX];
};

// the same formulas as unary.cu, so the fused chain reproduces the unfused kernels bit for bit
static __device__ __forceinline__ float fc_unary(const int32_t u, const float x) {
    switch (u) {
        case GGML_UNARY_OP_SIGMOID:    return 1.0f / (1.0f + expf(-x));
        case GGML_UNARY_OP_SILU:       return x / (1.0f + expf(-x));
        case GGML_UNARY_OP_SOFTPLUS:   return (x > 20.0f) ? x : logf(1.0f + expf(x));
        case GGML_UNARY_OP_EXP:        return expf(x);
        case GGML_UNARY_OP_TANH:       return tanhf(x);
        case GGML_UNARY_OP_RELU:       return fmaxf(x, 0);
        case GGML_UNARY_OP_NEG:        return -x;
        case GGML_UNARY_OP_ABS:        return fabsf(x);
        case GGML_UNARY_OP_GELU_QUICK: return x * (1.0f / (1.0f + expf(-1.702f * x)));
        default:                       return x;
    }
}

static __device__ __forceinline__ float fc_read(const char * base, const int64_t * ne, const int64_t * nb,
                                                const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) {
    const int64_t off = (i0 % ne[0]) * nb[0] + (i1 % ne[1]) * nb[1] + (i2 % ne[2]) * nb[2] + (i3 % ne[3]) * nb[3];
    return *(const float *) (base + off);
}

static __global__ void strixllama_chain_kernel(const strixllama_chain_params p) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.n) {
        return;
    }
    int64_t t = idx;
    const int64_t i0 = t % p.out_ne[0]; t /= p.out_ne[0];
    const int64_t i1 = t % p.out_ne[1]; t /= p.out_ne[1];
    const int64_t i2 = t % p.out_ne[2]; t /= p.out_ne[2];
    const int64_t i3 = t;

    float x = fc_read(p.in, p.in_ne, p.in_nb, i0, i1, i2, i3);
#pragma unroll
    for (int k = 0; k < STRIX_CHAIN_MAX; ++k) {
        if (k >= p.n_steps) {
            break;
        }
        const strixllama_chain_step & s = p.steps[k];
        switch (s.op) {
            case FC_SCALE: x = s.p0 * x + s.p1; break;
            case FC_CLAMP: x = x < s.p0 ? s.p0 : (x > s.p1 ? s.p1 : x); break;
            case FC_UNARY: x = fc_unary(s.unary, x); break;
            default: {
                const float y = fc_read(s.other, s.one, s.onb, i0, i1, i2, i3);
                const float a = s.chain_is_src1 ? y : x;
                const float b = s.chain_is_src1 ? x : y;
                switch (s.op) {
                    case FC_ADD: x = a + b; break;
                    case FC_SUB: x = a - b; break;
                    case FC_MUL: x = a * b; break;
                    default:     x = a / b; break;
                }
            }
        }
    }
    p.out[idx] = x;
}

// ---------------------------------------------------------------------------------------------------
// matcher

static bool fc_unary_ok(const ggml_tensor * t) {
    switch (ggml_get_unary_op(t)) {
        case GGML_UNARY_OP_SIGMOID: case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_SOFTPLUS: case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_GELU_QUICK:
            return true;
        default:
            return false;
    }
}

static int fc_op_of(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_SCALE: return FC_SCALE;
        case GGML_OP_CLAMP: return FC_CLAMP;
        case GGML_OP_UNARY: return fc_unary_ok(t) ? FC_UNARY : -1;
        case GGML_OP_ADD:   return FC_ADD;
        case GGML_OP_SUB:   return FC_SUB;
        case GGML_OP_MUL:   return FC_MUL;
        case GGML_OP_DIV:   return FC_DIV;
        default:            return -1;
    }
}

static bool fc_node_ok(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || fc_op_of(t) < 0) {
        return false;
    }
    if (t->src[0] == nullptr || t->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (fc_op_of(t) >= FC_ADD && (t->src[1] == nullptr || t->src[1]->type != GGML_TYPE_F32)) {
        return false;
    }
    return true;
}

// the tensor a chain node reads for its running value may be the previous node itself or a
// RESHAPE/contiguous VIEW alias of it; return the underlying node
static const ggml_tensor * fc_unalias(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || (t->op == GGML_OP_VIEW && t->view_offs == 0 && ggml_is_contiguous(t)))) {
        t = t->view_src ? t->view_src : t->src[0];
    }
    return t;
}

static bool fc_is_alias_or_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE || ggml_is_empty(t);
}

static void fc_set_operand(strixllama_chain_step & s, const ggml_tensor * o) {
    // a REPEAT consumed only by this node is folded into a broadcast read of its source
    s.other = (const char *) o->data;
    for (int d = 0; d < 4; ++d) { s.one[d] = o->ne[d]; s.onb[d] = o->nb[d]; }
}

static bool fc_enabled() {
    static const bool on = []() {
        const char * e = getenv("STRIX_CHAIN_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

int ggml_cuda_strixllama_chain_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!fc_enabled()) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    if ((first->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !fc_node_ok(first)) {
        return 0;
    }

    strixllama_chain_params p = {};
    // the chain starts with the first node's src0 as the running value (src1, if any, is the operand)
    const ggml_tensor * in = first->src[0];
    p.in = (const char *) in->data;
    for (int d = 0; d < 4; ++d) { p.in_ne[d] = in->ne[d]; p.in_nb[d] = in->nb[d]; }

    int n_steps = 0;
    const ggml_tensor * reads[STRIX_CHAIN_MAX] = {};
    // REPEAT nodes inside the chain are skipped and read through to their source. A REPEAT that ran
    // BEFORE the chain must be read as is: its source's lifetime ended at the REPEAT, and the
    // allocator may already have handed that memory to another tensor (the PLE value repeat).
    const ggml_tensor * folded[STRIX_CHAIN_MAX] = {};
    int n_folded = 0;
    auto fc_fold = [&](const ggml_tensor * o) -> const ggml_tensor * {
        if (o->op != GGML_OP_REPEAT) { return o; }
        for (int k = 0; k < n_folded; ++k) { if (folded[k] == o) { return o->src[0]; } }
        return o;
    };
    auto add_step = [&](const ggml_tensor * node, bool chain_is_src1) {
        strixllama_chain_step & s = p.steps[n_steps++];
        s.op = fc_op_of(node);
        s.unary = node->op == GGML_OP_UNARY ? (int32_t) ggml_get_unary_op(node) : 0;
        s.chain_is_src1 = chain_is_src1 ? 1 : 0;
        s.p0 = s.p1 = 0.0f;
        if (node->op == GGML_OP_SCALE || node->op == GGML_OP_CLAMP) {
            memcpy(&s.p0, (const float *) node->op_params + 0, sizeof(float));
            memcpy(&s.p1, (const float *) node->op_params + 1, sizeof(float));
        }
        s.other = nullptr;
        if (s.op >= FC_ADD) {
            const ggml_tensor * o = fc_fold(chain_is_src1 ? node->src[0] : node->src[1]);
            reads[n_steps - 1] = o;
            fc_set_operand(s, o);
        }
    };

    add_step(first, false);
    const ggml_tensor * prev = first;
    int prev_idx = i;
    int last_idx = i;

    for (int j = i + 1; j < cgraph->n_nodes && n_steps < STRIX_CHAIN_MAX; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (fc_is_alias_or_noop(node)) {
            // aliases of the running value are seen through; unrelated views cost nothing to skip
            continue;
        }
        if (node->op == GGML_OP_REPEAT) {
            static const bool no_repeat = getenv("STRIX_CHAIN_NO_REPEAT") != nullptr;
            if (no_repeat) {
                break;
            }
            // only a REPEAT that the very next chain node consumes alone can be folded
            if (j + 1 >= cgraph->n_nodes || ggml_node_get_use_count(cgraph, j) != 1 || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                break;
            }
            const ggml_tensor * nx = cgraph->nodes[j + 1];
            if (!fc_node_ok(nx) || fc_op_of(nx) < FC_ADD || (nx->src[0] != node && nx->src[1] != node)) {
                break;
            }
            if (!ggml_is_contiguous(node->src[0]) && node->src[0]->type != GGML_TYPE_F32) {
                break;
            }
            if (n_folded >= STRIX_CHAIN_MAX) {
                break;
            }
            folded[n_folded++] = node;
            continue;   // the consumer is handled on the next iteration and folds this REPEAT
        }
        if (!fc_node_ok(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            break;
        }
        const bool via0 = fc_unalias(node->src[0]) == prev;
        const bool via1 = node->src[1] != nullptr && fc_unalias(node->src[1]) == prev;
        if (!via0 && !via1) {
            break;
        }
        if (via0 && via1) {
            break;      // x op x: not a chain
        }
        // the running value must feed only this node (through at most an alias chain)
        if (ggml_node_get_use_count(cgraph, prev_idx) != 1 || (prev->flags & GGML_TENSOR_FLAG_OUTPUT) || prev->view_src) {
            break;
        }
        const ggml_tensor * link = via0 ? node->src[0] : node->src[1];
        bool alias_ok = true;
        for (const ggml_tensor * a = link; a != prev; a = a->view_src ? a->view_src : a->src[0]) {
            // every alias between prev and this node must itself be consumed only here
            int aidx = -1;
            for (int q = prev_idx + 1; q < j; ++q) { if (cgraph->nodes[q] == a) { aidx = q; break; } }
            if (aidx < 0 || ggml_node_get_use_count(cgraph, aidx) != 1 || (a->flags & GGML_TENSOR_FLAG_OUTPUT)) { alias_ok = false; break; }
        }
        if (!alias_ok) {
            break;
        }
        // shapes may only grow by broadcasting along the chain
        if (!ggml_can_repeat(link, node)) {
            break;
        }
        const ggml_tensor * other = via0 ? node->src[1] : node->src[0];
        if (other && !ggml_can_repeat(fc_fold(other), node)) {
            break;
        }
        // the operand must be memory that exists: not a chain node (never written) nor a view of one
        if (other) {
            const ggml_tensor * root = fc_fold(other);
            while (root && root->view_src) { root = root->view_src; }
            bool in_chain = false;
            for (int q = i; q <= prev_idx && !in_chain; ++q) {
                const ggml_tensor * c = cgraph->nodes[q];
                if (c == root || c == other) { in_chain = true; }
            }
            if (in_chain) {
                break;
            }
        }
        add_step(node, !via0);
        prev = node; prev_idx = j; last_idx = j;
    }

    if (n_steps < 2) {
        return 0;
    }
    const ggml_tensor * out = cgraph->nodes[last_idx];

    // the allocator only guarantees the chain's inputs stay valid until the node that read them ran;
    // once the intermediates are elided, the fused output may have been placed over any of them. Refuse
    // unless a read is exactly the output itself (same pointer and shape: each thread reads its own
    // element before writing it).
    auto overlaps = [&](const ggml_tensor * r) {
        const int64_t a0 = (int64_t) r->data, a1 = a0 + (int64_t) ggml_nbytes(r);
        const int64_t b0 = (int64_t) out->data, b1 = b0 + (int64_t) ggml_nbytes(out);
        if (a1 <= b0 || b1 <= a0) {
            return false;
        }
        return !(r->data == out->data && ggml_are_same_shape(r, out) && ggml_is_contiguous(r));
    };
    if (overlaps(in)) {
        return 0;
    }
    for (int k = 0; k < n_steps; ++k) {
        if (reads[k] && overlaps(reads[k])) {
            return 0;
        }
    }
    // the fused kernel pays four modulo reads per operand per element; that wins while the chain is
    // launch-bound (decode, small verification batches) and loses to the specialised kernels on the
    // multi-million-element tensors of a deep prefill (55K-token prefill: 968 -> 932 t/s unbounded)
    static const int64_t max_elems = []() {
        const char * e = getenv("STRIX_CHAIN_MAX_ELEMS");
        return e ? (int64_t) atoll(e) : (int64_t) 262144;
    }();
    if (ggml_nelements(out) > max_elems) {
        return 0;
    }
    p.out = (float *) out->data;
    for (int d = 0; d < 4; ++d) { p.out_ne[d] = out->ne[d]; }
    p.n = ggml_nelements(out);
    p.n_steps = n_steps;

    // bisection aid: STRIX_CHAIN_ONLY=FIRSTOP-LASTOP fuses only chains with that signature
    if (const char * only = getenv("STRIX_CHAIN_ONLY")) {
        char sig[64];
        snprintf(sig, sizeof(sig), "%s-%s", ggml_op_name(first->op), ggml_op_name(out->op));
        if (strcmp(sig, only) != 0) {
            return 0;
        }
    }

    static unsigned shown = 0;
    if (getenv("LLAMA_GRAPH_TRACE") && shown++ < 12) {
        fprintf(stderr, "CHAIN_FUSE: %d nodes [%s '%s' .. %s '%s'] in=[%lld,%lld,%lld,%lld] out=[%lld,%lld,%lld,%lld]\n",
                n_steps, ggml_op_name(first->op), first->name, ggml_op_name(out->op), out->name,
                (long long) in->ne[0], (long long) in->ne[1], (long long) in->ne[2], (long long) in->ne[3],
                (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2], (long long) out->ne[3]);
        for (int k = 0; k < n_steps; ++k) {
            const strixllama_chain_step & st = p.steps[k];
            fprintf(stderr, "   step %d: op %d unary %d chain_is_src1 %d p=(%g,%g) other=%p one=[%lld,%lld,%lld,%lld] onb=[%lld,%lld,%lld,%lld]\n",
                    k, st.op, st.unary, st.chain_is_src1, st.p0, st.p1, (const void *) st.other,
                    (long long) st.one[0], (long long) st.one[1], (long long) st.one[2], (long long) st.one[3],
                    (long long) st.onb[0], (long long) st.onb[1], (long long) st.onb[2], (long long) st.onb[3]);
        }
    }

    const int64_t blocks = (p.n + 255) / 256;

    // STRIX_CHAIN_VERIFY=1 (run with GGML_CUDA_DISABLE_GRAPHS=1): compute the chain node by node into
    // its real buffers first, keep that result, restore every operand the fused kernel reads (the plain
    // nodes may have written intermediates in place over them), then run the fused kernel and compare.
    static const bool verify = getenv("STRIX_CHAIN_VERIFY") != nullptr;
    std::vector<float> plain;
    std::vector<std::vector<char>> snaps;
    std::vector<const ggml_tensor *> snap_t;
    if (verify) {
        // the producers of the operands are still in flight on the (non-blocking) compute stream
        CUDA_CHECK(cudaDeviceSynchronize());
        snap_t.push_back(in);
        for (int k = 0; k < n_steps; ++k) { if (reads[k]) { snap_t.push_back(reads[k]); } }
        for (const ggml_tensor * t : snap_t) {
            snaps.emplace_back(ggml_nbytes(t));
            CUDA_CHECK(cudaMemcpy(snaps.back().data(), t->data, ggml_nbytes(t), cudaMemcpyDeviceToHost));
        }
        for (int q = i; q <= last_idx; ++q) {
            ggml_tensor * nq = cgraph->nodes[q];
            if (fc_is_alias_or_noop(nq)) { continue; }
            if (!ggml_cuda_compute_forward(ctx, nq)) { fprintf(stderr, "CHAIN_VERIFY: cannot recompute %s", nq->name); fputc(10, stderr); }
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        plain.resize(p.n);
        CUDA_CHECK(cudaMemcpy(plain.data(), out->data, p.n * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < snap_t.size(); ++k) {
            CUDA_CHECK(cudaMemcpy((void *) snap_t[k]->data, snaps[k].data(), snaps[k].size(), cudaMemcpyHostToDevice));
        }
        // the restores go through the null stream; the fused kernel runs on a non-blocking stream
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    strixllama_chain_kernel<<<(unsigned) blocks, 256, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());

    if (verify) {
        std::vector<float> fused(p.n);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(fused.data(), out->data, p.n * sizeof(float), cudaMemcpyDeviceToHost));
        int64_t bad = 0, first_bad = -1; float maxd = 0.0f;
        for (int64_t e = 0; e < p.n; ++e) {
            const float d = fabsf(fused[e] - plain[e]);
            if (d != d || d > 0) { if (first_bad < 0) { first_bad = e; } ++bad; if (d > maxd) { maxd = d; } }
        }
        static unsigned reports = 0, exact = 0;
        if (!bad) { ++exact; }
        if (bad || reports++ < 40) {
            fprintf(stderr, "CHAIN_VERIFY: %u exact so far", exact); fputc(10, stderr);
            fprintf(stderr, "CHAIN_VERIFY: [%s .. %s] %lld/%lld elements differ, max |d| %g%s\n", first->name, out->name,
                    (long long) bad, (long long) p.n, maxd, bad ? "" : " (exact)");
            if (bad) {
                fprintf(stderr, "   first bad at %lld: fused %g plain %g; neighbours fused %g %g plain %g %g\n", (long long) first_bad,
                        fused[first_bad], plain[first_bad], first_bad + 1 < p.n ? fused[first_bad + 1] : 0.f, first_bad + 2 < p.n ? fused[first_bad + 2] : 0.f,
                        first_bad + 1 < p.n ? plain[first_bad + 1] : 0.f, first_bad + 2 < p.n ? plain[first_bad + 2] : 0.f);
            }
        }
    }
    return last_idx - i;
}
