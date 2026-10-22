#include "gdn-conv.cuh"
#include "unary.cuh"
#include "norm-gated.cuh"
#include "gated_delta_net.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

// STRIX_GDN_CONV_HALO=1: the halo copies even when the conv's output is not x's memory (a test of that path)
static bool gdn_conv_force_halo() { static const bool v = getenv("STRIX_GDN_CONV_HALO") && atoi(getenv("STRIX_GDN_CONV_HALO")) != 0; return v; }
static bool gdn_conv_enabled() { static const int v = getenv("LLAMA_GDN_CONV") ? atoi(getenv("LLAMA_GDN_CONV")) : 0; return v != 0; }

// strixllama: STRIX_GDN_CONV_NORM_LAZY=0 - the deferred-rollback net reads the unfused norms' q and k (-1: the STRIX_AB side)
static bool gdn_conv_norm_lazy() {
    static const int v = getenv("STRIX_GDN_CONV_NORM_LAZY") ? atoi(getenv("STRIX_GDN_CONV_NORM_LAZY")) : 1;
    return v > 0 || (v < 0 && ggml_cuda_strix_ab());
}
// strixllama: STRIX_GDN_CONV_TAIL=0 - the decode batch's state copies stay CPY nodes (-1: the STRIX_AB side)
static bool gdn_conv_tail_on() {
    static const int v = getenv("STRIX_GDN_CONV_TAIL") ? atoi(getenv("STRIX_GDN_CONV_TAIL")) : 1;
    return v > 0 || (v < 0 && ggml_cuda_strix_ab());
}

// the state copies a kernel of the conv does (ggml_cuda_gdn_conv_match::n_tail): slot s gets the concat's columns
// col[s] .. col[s] + 2 of every channel, laid out as the CPY of that [3, C] view into a contiguous row writes them
struct gdn_conv_tails {
    float * dst[8];
    int     col[8];
    int     n;
};

static gdn_conv_tails gdn_conv_tails_of(const ggml_cuda_gdn_conv_match & m) {
    gdn_conv_tails tl = {};
    tl.n = m.n_tail;
    for (int s = 0; s < m.n_tail; ++s) { tl.dst[s] = m.tail_dst[s]; tl.col[s] = m.tail_col[s]; }
    return tl;
}

// a thread's channel, before the kernel writes anything (its output may be x's own memory)
static __device__ __forceinline__ void gdn_conv_write_tails(const gdn_conv_tails & tl, const float * __restrict__ state,
        const float * x, const int C, const int c) {
    for (int s = 0; s < tl.n; ++s) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int col = tl.col[s] + k;
            tl.dst[s][c * 3 + k] = col < 3 ? state[c * 3 + col] : x[(size_t) (col - 3) * C + c];
        }
    }
}

// tail-only materialization of the concat: columns [tail_from, T+3) of every channel row
static __global__ void gdn_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int tail_from, const int row_stride) {
    const int ncols = T + 3 - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) return;
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < 3) ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c];
}

// strixllama: the three rows of x before each token block but the first, copied before the conv runs. The allocator frees
// x after the concat, the last node that reads it in the graph, and mostly gives its block to the conv's output, which the
// SiLU then shares: the conv writes the rows it reads. Within a block a channel's rows are read before they are written;
// a block's first three come from the block before, which may have written them already
template <int TT>
static __global__ void __launch_bounds__(256) gdn_conv_halo_kernel(const float * __restrict__ x, float * __restrict__ halo, const int C) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y + 1;
#pragma unroll
    for (int k = 0; k < 3; ++k) halo[((size_t) (b - 1) * 3 + k) * C + c] = x[(size_t) (b * TT - 3 + k) * C + c];
}

// the window a token block starts with: rows t0 - 3 .. t0 of channel c, the state's before the first token, else x's or
// (halo, when the conv writes x's memory) their copies
static __device__ __forceinline__ float gdn_conv_start(const float * __restrict__ state, const float * x,
        const float * __restrict__ halo, const int C, const int c, const int t0, const int k) {
    const int jp = t0 + k;
    if (jp < 3) return state[c * 3 + jp];
    if (halo != nullptr && k < 3) return halo[((size_t) (blockIdx.y - 1) * 3 + k) * C + c];
    return x[(size_t) (jp - 3) * C + c];
}

template <bool apply_silu, int TT>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ state, const float * x,
        const float * __restrict__ w, float * y, const int C, const int T, const float * __restrict__ halo,
        const gdn_conv_tails tails = {}) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) return;
    if (tails.n > 0 && blockIdx.y == 0) gdn_conv_write_tails(tails, state, x, C, c);
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) xw[k] = gdn_conv_start(state, x, halo, C, c, t0, k);
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
        sumf += b;
        y[(size_t) t * C + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4; xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
}

static __device__ __forceinline__ float gdn_xor_tree(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor(v, off); }
    return v;
}

// strixllama: the conv and its SiLU as gdn_conv_direct_kernel, and for the q and k channels (the first 2 nq) the L2 norm
// the graph builds after it - rms_norm(eps / 128) then scale(1 / sqrt(128)), norm-gated.cu's rms_rows_f32<false, false,
// true> - in the same pass, the normed rows written where the raw ones would have gone and read there by the net
// (ggml_cuda_gdn_qk_from). Two norms read 67 MB and wrote 67 MB each at 8K tokens a layer (~1 ms each). A round of TB
// tokens puts its SiLU rows in LDS and then runs rms_rows_f32's per-row code on them as written there, a wave per row
// (token, head): the same operations in the same order. Bitwise only so: the partial sums taken where the channels are,
// one wave per 32 of them, came out in the same order on paper and differed in the last bit in ~5% of the rows
template <int TT, int TB>
static __global__ void __launch_bounds__(256) gdn_conv_norm_kernel(const float * __restrict__ state, const float * x,
        const float * __restrict__ w, float * y, const int C, const int T, const int nq, const int ncols, const float eps,
        const float post_scale, const float post_bias, const float * __restrict__ halo, const gdn_conv_tails tails = {}) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (tails.n > 0 && blockIdx.y == 0) gdn_conv_write_tails(tails, state, x, C, c);
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) xw[k] = gdn_conv_start(state, x, halo, C, c, t0, k);
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    if ((int) blockIdx.x * 256 >= 2 * nq) {
        for (int t = t0; t < tend; ++t) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
            sumf += b;
            y[(size_t) t * C + c] = ggml_cuda_op_silu_single(sumf);
            xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
            const int jp = t + 4; xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
        }
        return;
    }
    __shared__ float rows[TB][256];   // the round's SiLU values: [token][channel of the block]
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int cb = blockIdx.x * 256;   // the block's first channel: q (< nq) or k
    for (int tb0 = t0; tb0 < tend; tb0 += TB) {
#pragma unroll
        for (int u = 0; u < TB; ++u) {
            const int t = tb0 + u;
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
            sumf += b;
            const float val = ggml_cuda_op_silu_single(sumf);
            rows[u][threadIdx.x] = val;
            xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
            const int jp = t + 4; xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
        }
        __syncthreads();
        // rms_rows_f32<false, false, true>'s row, verbatim but for where x and dst are
        for (int r = wave; r < 2 * TB; r += 8) {
            const int u = r >> 1, hh = r & 1, t = tb0 + u;
            if (t >= tend) continue;
            const float * xr = rows[u] + 128 * hh;
            // the SiLU's own row: its raw q and k values have no other reader. Rows up to tb0 + TB - 1 were read before the
            // barrier, the next one is not written in this round, and the next block's first three come from the halo copy
            float * dst = y + (size_t) t * C + cb + 128 * hh;
            float part[8];
#pragma unroll
            for (int wv = 0; wv < 8; ++wv) {
                const int col = 32 * wv + lane;
                const float xi = col < ncols ? xr[col] : 0.0f;
                part[wv] = gdn_xor_tree(xi * xi);
            }
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < 8; ++wv) { v = lane == wv ? part[wv] : v; }
            const float tmp = gdn_xor_tree(v);
            const float mean = tmp / ncols;
            const float scale = rsqrtf(mean + eps);
#pragma unroll
            for (int wv = 0; wv < 8; ++wv) {
                const int col = 32 * wv + lane;
                if (col < ncols) {
                    float tv;
                    tv = scale * xr[col];
                    tv = post_scale * tv + post_bias;
                    dst[col] = tv;
                }
            }
        }
        __syncthreads();
    }
}

// the bytes of a and b overlap
static bool gdn_conv_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    if (a0 == nullptr || b0 == nullptr) return false;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) return false;
    if (tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t C = x->ne[0], T = x->ne[1];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) return false;
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) return false;
    // strixllama: STRIX_GDN_CONV_MIN_T - the fewest tokens the fused conv takes (a decode step: 1)
    static const int min_t = getenv("STRIX_GDN_CONV_MIN_T") ? atoi(getenv("STRIX_GDN_CONV_MIN_T")) : 1;
    if (T < min_t || C % 256 != 0) return false;
    if (i + 1 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) return false;
    int64_t tail_from = T + 3; int conv = -1;
    int cpys[8]; int ncpy = 0; bool cont_reader = false;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cc || t->src[s]->view_src == cc) reads = true;
        if (!reads) continue;
        if (t->op == GGML_OP_VIEW) {
            if (t->ne[0] != 3 || t->ne[1] != C || t->nb[1] != cc->nb[1] || t->view_offs % sizeof(float)) return false;
            const int64_t col = (int64_t) (t->view_offs / sizeof(float)); if (col < 0 || col + 3 > T + 3) return false;
            tail_from = std::min(tail_from, col); continue;
        }
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) { cont_reader = true; continue; }
        // strixllama: the state update copies the tail straight out of the concat now, without a cont first
        if (t->op == GGML_OP_CPY && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) {
            if (ncpy < 8) cpys[ncpy] = n;
            ++ncpy;
            continue;
        }
        return false;
    }
    if (conv < 0) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) return false;
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || !ggml_is_contiguous(cv)) return false;
    if (conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) return false;
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) { const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cv || t->src[s]->view_src == cv) return false; }
    // strixllama: the kernel reads x and the state rows at the conv's node, and the allocator frees both after the concat,
    // the last node that reads them in the graph. The SiLU's output often is x's own block (the halos are copied first,
    // gdn_conv_halo_kernel); anywhere else on x, or on the state rows (qwen4exp keeps them allocated), it would be read
    // while written. Measured 10-12 at 8K and 2K tokens: on x's block in most layers before the state rows were kept, on
    // nothing since
    const bool alias = su->data != nullptr && su->data == x->data && ggml_nbytes(su) == ggml_nbytes(x);
    if ((gdn_conv_overlap(su, x) && !alias) || gdn_conv_overlap(su, st)) return false;
    m.alias = alias;
    m.concat_idx = i; m.conv_idx = conv; m.x = x; m.state = st; m.concat = cc; m.w = w; m.conv_out = cgraph->nodes[conv + 1];
    m.C = C; m.T = T; m.tail_from = tail_from; m.silu = true;
    // a decode batch: the copies of the tail into the cache rows, each a [3, C] view of the concat to a contiguous row
    // its CPY's only effect, done by the conv's kernel instead (ggml_cuda_gdn_conv_tail_skip)
    m.n_tail = 0;
    if (gdn_conv_tail_on() && T <= 8 && !cont_reader && ncpy > 0 && ncpy <= 8) {
        bool ok = true;
        for (int q = 0; q < ncpy && ok; ++q) {
            const ggml_tensor * cp = cgraph->nodes[cpys[q]];
            const ggml_tensor * sv = cp->src[0], * dv = cp->src[1];
            ok = sv->ne[0] == 3 && sv->ne[1] == C && sv->ne[2] == 1 && sv->ne[3] == 1 && sv->nb[0] == sizeof(float) &&
                 sv->nb[1] == cc->nb[1] && sv->view_offs % sizeof(float) == 0 && sv->view_offs / sizeof(float) + 3 <= (size_t) (T + 3) &&
                 dv != nullptr && dv->type == GGML_TYPE_F32 && dv->data != nullptr && ggml_nelements(dv) == 3 * C &&
                 ggml_is_contiguous(dv) && cp->type == GGML_TYPE_F32 && ggml_node_get_use_count(cgraph, cpys[q]) == 0;
            if (ok) {
                m.tail_dst[q] = (float *) dv->data;
                m.tail_col[q] = (int) (sv->view_offs / sizeof(float));
                m.tail_cpy[q] = cp;
            }
        }
        // nothing after the conv reads the concat (it is not written at all)
        for (int n = conv + 1; n < cgraph->n_nodes && ok; ++n) {
            const ggml_tensor * t = cgraph->nodes[n];
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                if (t->src[s] == cc || t->src[s]->view_src == cc) ok = false;
            }
        }
        if (ok) m.n_tail = ncpy;
    }
    return true;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) { return gdn_conv_check(cgraph, i, m); }

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) return false;
    const ggml_tensor * cc = cgraph->nodes[j]->src[0];
    for (int i = j - 1; i >= 0 && i >= j - 64; --i) if (cgraph->nodes[i] == cc) return gdn_conv_check(cgraph, i, m) && m.conv_idx == j;
    return false;
}

// a view of the conv's rows [C, T] that stays inside each row's channels from `from` bytes on (the v view and its reshapes)
static bool gdn_conv_view_past(const ggml_tensor * v, int64_t C, size_t from) {
    const size_t row = (size_t) C * sizeof(float);
    if (v->nb[0] != sizeof(float) || v->view_offs % row < from) return false;
    size_t inner = sizeof(float) * v->ne[0];   // the bytes the dimensions inside a row span from the view's start
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (v->ne[d] <= 1 || v->nb[d] % row == 0) continue;   // whole rows apart
        if (v->nb[d] >= row) return false;
        inner += v->nb[d] * (v->ne[d] - 1);
    }
    return v->view_offs % row + inner <= row;
}

// after the SiLU: VIEW (q or k rows of it), RMS_NORM of that view, SCALE of the norm - the pair norm-gated.cu would run as
// rms_rows_f32<false, false, true> - for q (offset 0) and k (offset nq channels), nothing else reading those channels
bool ggml_cuda_gdn_conv_norm_match(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m, ggml_cuda_gdn_conv_norm & n) {
    static const bool on = !getenv("STRIX_GDN_CONV_NORM") || atoi(getenv("STRIX_GDN_CONV_NORM")) != 0;
    if (!on || !m.silu || m.conv_out == nullptr) return false;
    const ggml_tensor * y = m.conv_out;
    const int js = m.conv_idx + 1;
    const ggml_tensor * views[2] = { nullptr, nullptr };
    ggml_cuda_norm_scale_match sm[2];
    int k = js + 1;
    for (int r = 0; r < 2; ++r, k += 3) {
        if (k + 2 >= cgraph->n_nodes) return false;
        const ggml_tensor * v = cgraph->nodes[k];
        if (v->op != GGML_OP_VIEW || v->view_src != y || cgraph->nodes[k + 1]->op != GGML_OP_RMS_NORM || cgraph->nodes[k + 1]->src[0] != v) return false;
        // any number of rows: the fused kernel's row code is rms_rows_f32's at every count
        if (ggml_cuda_norm_scale_match_at(cgraph, k + 1, sm[r], 0) <= 0 || sm[r].dst != cgraph->nodes[k + 2] || sm[r].x != v) return false;
        if (ggml_node_get_use_count(cgraph, k) != 1) return false;   // a view: has_n_uses refuses views, the scan below covers the rest
        if (v->ne[0] != 128 || v->nb[0] != sizeof(float) || v->nb[1] != 128 * sizeof(float) || v->ne[2] != m.T || v->ne[3] != 1 ||
            v->nb[2] != (size_t) m.C * sizeof(float) || v->view_offs % sizeof(float)) return false;
        const ggml_tensor * d = sm[r].dst;
        if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || !ggml_are_same_shape(d, v)) return false;
        views[r] = v;
    }
    const ggml_tensor * vq = views[0]->view_offs == 0 ? views[0] : views[1];
    const ggml_tensor * vk = vq == views[0] ? views[1] : views[0];
    const int64_t nq = vq->ne[0] * vq->ne[1];
    if (vq->view_offs != 0 || vk->view_offs != (size_t) nq * sizeof(float) || vk->ne[1] != vq->ne[1]) return false;
    if ((2 * nq) % 256 != 0 || 2 * nq > m.C) return false;
    const ggml_cuda_norm_scale_match & a = sm[0], & b = sm[1];
    if (a.eps != b.eps || a.scale != b.scale || a.bias != b.bias) return false;
    // the raw q and k channels of the SiLU are not written: no other node may read them
    const size_t qk_bytes = (size_t) (2 * nq) * sizeof(float);
    for (int t = js + 1; t < cgraph->n_nodes; ++t) {
        const ggml_tensor * node = cgraph->nodes[t];
        if (node == views[0] || node == views[1]) continue;
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == views[0] || src == views[1]) {
                if (node->op != GGML_OP_RMS_NORM) return false;
                continue;
            }
            if (src == y && node->view_src != y) return false;
            if (src->view_src == y && !gdn_conv_view_past(src, m.C, qk_bytes)) return false;
        }
        if (node->view_src == y && !gdn_conv_view_past(node, m.C, qk_bytes)) return false;
    }
    n.q = vq == views[0] ? sm[0].dst : sm[1].dst;
    n.k = vq == views[0] ? sm[1].dst : sm[0].dst;
    // their only reader: the net, plain or with deferred rollback (mode 1; mode 2 is the replay alone), as its q and k
    n.gdn = nullptr;
    for (int t = js + 7; t < cgraph->n_nodes && t < js + 64; ++t) {
        const ggml_tensor * g = cgraph->nodes[t];
        if (g->op == GGML_OP_GATED_DELTA_NET && (g->src[0] == n.q || g->src[1] == n.q)) {
            const int mode = ggml_get_op_params_i32(g, 1);
            if (g->src[0] != n.q || g->src[1] != n.k || (mode != 0 && (mode != 1 || !gdn_conv_norm_lazy()))) return false;
            n.gdn = g;
            break;
        }
    }
    if (n.gdn == nullptr) return false;
    for (int t = 0; t < 2; ++t) {
        const int idx = js + 3 + 3 * t;   // the SCALE nodes
        if (ggml_node_get_use_count(cgraph, idx) != 1 || (cgraph->nodes[idx]->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
    }
    n.nq = nq; n.eps = a.eps; n.scale = a.scale; n.bias = a.bias; n.vq = vq; n.vk = vk;
    n.last = js + 6;
    return true;
}

void ggml_cuda_gdn_conv_norm_run(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m, const ggml_cuda_gdn_conv_norm & n) {
    static unsigned hits = 0;
    if (hits++ == 0) fprintf(stderr, "GDN_CONV_NORM fused: C=%lld T=%lld nq=%lld\n", (long long) m.C, (long long) m.T, (long long) n.nq);
    constexpr int TT = 128, TB = 8;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    ggml_cuda_pool_alloc<float> halo_buf(ctx.pool());
    const float * halo = nullptr;
    if ((m.alias || gdn_conv_force_halo()) && grid.y > 1) {
        halo_buf.alloc((size_t) (grid.y - 1) * 3 * m.C);
        gdn_conv_halo_kernel<TT><<<dim3(grid.x, grid.y - 1), 256, 0, ctx.stream()>>>((const float *) m.x->data, halo_buf.get(), (int) m.C);
        CUDA_CHECK(cudaGetLastError());
        halo = halo_buf.get();
    }
    // STRIX_GDN_CONV_NORM_CHECK=1 (with GGML_CUDA_DISABLE_GRAPHS=1): the unfused kernels into scratch first (with the output
    // on x's memory the fused pass overwrites x), then compared bitwise with the fused rows
    static const bool check = getenv("STRIX_GDN_CONV_NORM_CHECK") != nullptr;
    static int checked = 0;
    const bool do_check = check && checked < 40;
    const int64_t ny = m.C * m.T, nqk = n.nq * m.T;
    ggml_cuda_pool_alloc<float> ry(ctx.pool()), rq(ctx.pool()), rk(ctx.pool());
    if (do_check) {
        ry.alloc(ny); rq.alloc(nqk); rk.alloc(nqk);
        gdn_conv_direct_kernel<true, 128><<<grid, 256, 0, ctx.stream()>>>(
            (const float *) m.state->data, (const float *) m.x->data, (const float *) m.w->data, ry.get(), (int) m.C, (int) m.T, halo);
        for (int r = 0; r < 2; ++r) {
            ggml_tensor xv = *(r ? n.vk : n.vq); xv.data = (char *) ry.get() + xv.view_offs;
            ggml_tensor dv = *(r ? n.k : n.q);   dv.data = r ? rk.get() : rq.get();
            ggml_cuda_norm_scale_match sm; sm.x = &xv; sm.dst = &dv; sm.eps = n.eps; sm.scale = n.scale; sm.bias = n.bias;
            ggml_cuda_op_norm_scale(ctx, sm);
        }
    }
    float * y = (float *) m.conv_out->data;
    // a decode batch: rounds of one token, the kernel's code a fraction of the eight-token round's (it is fetched anew at
    // every launch between the large products); the same per-row operations
    auto kernel = m.T <= 8 ? gdn_conv_norm_kernel<TT, 1> : gdn_conv_norm_kernel<TT, TB>;
    kernel<<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const float *) m.w->data, y, (int) m.C, (int) m.T, (int) n.nq, 128, n.eps, n.scale, n.bias, halo, gdn_conv_tails_of(m));
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_gdn_qk_from(n.gdn, y, y + n.nq, 128, m.C, m.C * m.T);
    if (do_check) {
        ++checked;
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        std::vector<float> ya(ny), yb(ny), b(nqk);
        CUDA_CHECK(cudaMemcpy(ya.data(), y, ny * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(yb.data(), ry.get(), ny * sizeof(float), cudaMemcpyDeviceToHost));
        for (int r = 0; r < 2; ++r) {
            CUDA_CHECK(cudaMemcpy(b.data(), r ? rk.get() : rq.get(), nqk * sizeof(float), cudaMemcpyDeviceToHost));
            int64_t bad = 0, first = -1;
            for (int64_t e = 0; e < nqk; ++e) {
                const float fa = ya[(e / n.nq) * m.C + r * n.nq + e % n.nq];
                if (memcmp(&fa, &b[e], 4) != 0) { if (first < 0) first = e; ++bad; }
            }
            fprintf(stderr, "CN_CHECK conv node %d %s: %lld of %lld differ (alias %d)\n", m.conv_idx, r ? "k" : "q", (long long) bad, (long long) nqk, (int) m.alias);
        }
        int64_t bad = 0;
        for (int64_t e = 0; e < ny; ++e) if (e % m.C >= 2 * n.nq && memcmp(&ya[e], &yb[e], 4) != 0) ++bad;
        fprintf(stderr, "CN_CHECK conv node %d v: %lld differ\n", m.conv_idx, (long long) bad);
    }
}

static std::vector<const ggml_tensor *> g_gdn_conv_tail_skip;

void ggml_cuda_gdn_conv_tail_reset() {
    g_gdn_conv_tail_skip.clear();
}

bool ggml_cuda_gdn_conv_tail_skip(const ggml_tensor * node) {
    if (node->op != GGML_OP_CPY || g_gdn_conv_tail_skip.empty()) return false;
    for (auto it = g_gdn_conv_tail_skip.begin(); it != g_gdn_conv_tail_skip.end(); ++it) {
        if (*it == node) { g_gdn_conv_tail_skip.erase(it); return true; }
    }
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    if (m.n_tail > 0) {   // the conv's kernel writes the copies, nobody reads the concat
        // the previous layer's copies were all reached before this concat (they read that layer's concat)
        g_gdn_conv_tail_skip.clear();
        for (int s = 0; s < m.n_tail; ++s) {
            g_gdn_conv_tail_skip.push_back(m.tail_cpy[s]);
        }
        return;
    }
    const int ncols = (int) (m.T + 3 - m.tail_from);
    if (ncols <= 0) return;
    const int64_t n = m.C * ncols;
    gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    ggml_cuda_pool_alloc<float> halo(ctx.pool());
    if ((m.alias || gdn_conv_force_halo()) && grid.y > 1) {
        halo.alloc((size_t) (grid.y - 1) * 3 * m.C);
        gdn_conv_halo_kernel<TT><<<dim3(grid.x, grid.y - 1), 256, 0, ctx.stream()>>>((const float *) m.x->data, halo.get(), (int) m.C);
        CUDA_CHECK(cudaGetLastError());
    }
    gdn_conv_direct_kernel<true, TT><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const float *) m.w->data, (float *) m.conv_out->data, (int) m.C, (int) m.T, (m.alias || gdn_conv_force_halo()) && grid.y > 1 ? halo.get() : nullptr,
        gdn_conv_tails_of(m));
    CUDA_CHECK(cudaGetLastError());
}
