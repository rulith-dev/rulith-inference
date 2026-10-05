#!/usr/bin/env python3
"""UD-Q4_K_XL's expert types in the prefill-sized MoE microbenchmark (STRIX_MOE_GLU_PERF), a way to pin
which recorded routing line it replays, and cases for the delta net's narrow-row norms (STRIX_NORM_PERF).

0.4.2 gave glu3 a Q4_K dequantization (WQ 12) and the routed plain kernel Q5_1 (WTYPE 4), which is
what UD-Q4_K_XL's experts are stored as, but make_test_cases_perf still built only IQ3_S / IQ4_NL /
IQ4_XS cases. The kernel that carries ~37% of that file's prefill could not be timed on its own.

Adds GGML_TYPE_Q4_K to the fused gate/up cases and GGML_TYPE_Q5_1 to the down cases, and lets
STRIX_MOE_GLU_TYPES name them (q4_k, q5_1) instead of only accepting "all". The default is unchanged:
IQ3_S gate/up and IQ4_NL down.

STRIX_MOE_IDS_LINE answers a second problem: init_recorded_ids walks its candidate list with a turn
counter, so a run that happens to iterate a different number of times averages a different set of
layers, and the case's time moves with the set (iq3_s at 8156 tokens read 20.2 and 25.2 ms that way).
Pinning one line, and the shuffle seed with it, makes an A/B comparable.

STRIX_NORM_PERF adds the delta net's narrow-row norms: the bare RMS_NORM shapes the prompt graph
charges 95.9 / 17.6 / 11.0 ms to for the same bytes an element, and the gated form qwen4exp actually
builds for its output norm (RMS_NORM, MUL(w), SIGMOID(z), MUL), which is a different kernel
instantiation and reads half the bytes again. Without both, a standalone number is not comparable
with the model's node.

Replay a recorded routing with STRIX_MOE_IDS_FILE, and pick the token count that routing was recorded
at: a dump line is used when its n_rows is a multiple of the test's token count, so a dump taken from
one prompt per length must be replayed at that exact count.

Each hunk is applied only if it is not there yet, so a later hunk can be added without reverting the
earlier ones.

Usage: python patches/apply_moe_glu_kq_perf.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
MARKER = 'MOE_GLU_KQ'

EDITS = {
    os.path.join(tree, "ggml", "src", "ggml-cuda", "norm-gated.cu"): [
        # the two fused-norm launches print their shape on the first two hits; the strides are what a
        # standalone case cannot reproduce, and the matchers check only x->nb[0], so print them too
        ('''    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "NORM_ROWS+SCALE fused: ncols=%d rows=%lld\\n", (int) x->ne[0], (long long) (x->ne[1]*x->ne[2]*x->ne[3]));
''',
         '''    // strixllama MOE_GLU_KQ: the strides too - a view's nb[1]/nb[2] decide this kernel's coalescing and
    // neither matcher checks them (only x->nb[0])
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "NORM_ROWS+SCALE fused: ncols=%d rows=%lld strides=%lld/%lld nb0=%lld\\n", (int) x->ne[0], (long long) (x->ne[1]*x->ne[2]*x->ne[3]), (long long) (x->nb[1]/4), (long long) (x->nb[2]/4), (long long) (x->nb[0]/4));
'''),
        ('''    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "NORM_ROWS%s fused: ncols=%d rows=%lld\\n", m.z ? "+GATE" : "", (int) x->ne[0], (long long) (x->ne[1]*x->ne[2]*x->ne[3]));
''',
         '''    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "NORM_ROWS%s fused: ncols=%d rows=%lld strides=%lld/%lld nb0=%lld xbuf=%s wbuf=%s\\n", m.z ? "+GATE" : "", (int) x->ne[0], (long long) (x->ne[1]*x->ne[2]*x->ne[3]), (long long) (x->nb[1]/4), (long long) (x->nb[2]/4), (long long) (x->nb[0]/4), ggml_backend_buffer_name(x->buffer), m.w ? ggml_backend_buffer_name(m.w->buffer) : "-");
'''),
    ],
    os.path.join(tree, "tests", "test-backend-ops.cpp"): [
        # 1. the case list
        ('''        const char * types = getenv("STRIX_MOE_GLU_TYPES");   // "iq3_s" by default; "all" adds the others
        const bool all = types && strcmp(types, "all") == 0;
        for (int n : ns) {
            for (ggml_type t : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS}) {
                if (t != GGML_TYPE_IQ3_S && !all) continue;
                test_cases.emplace_back(new test_moe_glu(t, 512, 10, 640, n, 2560));
            }
            for (ggml_type t : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0}) {
                if (t != GGML_TYPE_IQ4_NL && !all) continue;
                test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 512, 10, false, 2560, n, 640));   // down
            }
        }
''',
         '''        // strixllama MOE_GLU_KQ: UD-Q4_K_XL's expert types too - Q4_K gate/up (the fused GLU, glu3 WQ 12) and
        // Q5_1 down (the routed plain kernel, WTYPE 4). STRIX_MOE_GLU_TYPES=all still means every type; a list
        // names them: iq3_s iq4_nl iq4_xs q4_k q5_1 q8_0. dflt is the type each loop includes with no list set.
        const char * types = getenv("STRIX_MOE_GLU_TYPES");   // "iq3_s" by default; "all" adds the others
        const bool all = types && strcmp(types, "all") == 0;
        auto wanted = [&](ggml_type t, const char * dflt) {
            const char * key = t == GGML_TYPE_Q4_K ? "q4_k" : ggml_type_name(t);
            if (all) return true;
            if (!types || !*types) return strcmp(ggml_type_name(t), dflt) == 0;
            return strstr(types, key) != nullptr;
        };
        for (int n : ns) {
            for (ggml_type t : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K}) {
                if (!wanted(t, "iq3_s")) continue;
                test_cases.emplace_back(new test_moe_glu(t, 512, 10, 640, n, 2560));
            }
            for (ggml_type t : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
                if (!wanted(t, "iq4_nl")) continue;
                test_cases.emplace_back(new test_mul_mat_id(t, GGML_TYPE_F32, 512, 10, false, 2560, n, 640));   // down
            }
        }
'''),
        # 2. pin the replayed routing line
        ('''    static size_t turn = 0;
    const std::vector<int> & v = *cand[turn++ % cand.size()];
''',
         '''    static size_t turn = 0;
    // strixllama MOE_GLU_KQ: STRIX_MOE_IDS_LINE pins which matching line every call replays. The turn
    // counter walks the list, so a run whose iteration count differs averages a different set of layers
    // and the case's time moves with the set (iq3_s at 8156 tokens read 20.2 and 25.2 ms this way).
    static const int pin = getenv("STRIX_MOE_IDS_LINE") ? atoi(getenv("STRIX_MOE_IDS_LINE")) : -1;
    const size_t idx = pin >= 0 ? (size_t) pin % cand.size() : turn++;
    const std::vector<int> & v = *cand[idx];
'''),
        # 3. the shuffle seed follows the line, so the same tokens land on the same experts every call
        ('''    std::default_random_engine rng(1234 + (unsigned) turn);
''',
         '''    std::default_random_engine rng(1234 + (unsigned) idx);
'''),
        # 4. the narrow-row norm cases
        ('''static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;
''',
         '''static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // strixllama MOE_GLU_KQ: the narrow-row norms of a prompt's pass, one case a shape. The delta net
    // norms q and k as 16 groups x T rows of 128, and its output norm as 48 of 128 (inner_size 6144).
    // A bare RMS_NORM of F32 takes ggml_cuda_rms_rows_plain (norm-gated.cu), one warp a row.
    // STRIX_NORM_PERF=2048 or a token list.
    if (const char * norm = getenv("STRIX_NORM_PERF")) {
        std::vector<int> ns = {2048};
        if (strchr(norm, ',') || atoi(norm) > 1) {
            ns.clear();
            for (const char * c = norm; *c; ) { ns.push_back(atoi(c)); c = strchr(c, ','); if (!c) break; ++c; }
        }
        for (int n : ns) {
            for (const std::array<int64_t, 4> sh : {std::array<int64_t, 4>{128, 48, n, 1},
                                                    std::array<int64_t, 4>{128, 16, n, 1},
                                                    std::array<int64_t, 4>{256, 24, n, 1}}) {
                test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, sh));
                test_cases.emplace_back(new test_norm_gated(sh));
                test_cases.emplace_back(new test_norm_gated(sh, 1e-6f, true));
            }
        }
        return test_cases;
    }
'''),
        # 5. the gated norm as the model builds it, so the standalone number is comparable
        ('''static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
''',
         '''// strixllama MOE_GLU_KQ: the delta net's gated output norm as qwen4exp builds it (build_norm_gated:
// RMS_NORM, MUL(w), SIGMOID(z), MUL), which is the pattern ggml_cuda_norm_gated_match_at fuses into one
// rms_rows_f32 launch with the gate on. -o NORM_GATED selects it. The bare RMS_NORM cases are a
// different instantiation (<false,false> against <true>) and read fewer bytes, so this is the one that
// compares with the model's node.
struct test_norm_gated : public test_case {
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool producer;   // build z with a MUL_MAT ahead of it, as build_norm_gated's z_2d is

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "NORM_GATED"; }
    bool run_whole_graph() override { return true; }
    std::string vars() override { return VARS_TO_STR3(ne, eps, producer); }

    test_norm_gated(std::array<int64_t, 4> ne = {128, 48, 2048, 1}, float eps = 1e-6f, bool producer = false)
        : ne(ne), eps(eps), producer(producer) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_tensor * z;
        ggml_set_param(x); ggml_set_name(x, "x");
        ggml_set_param(w); ggml_set_name(w, "w");
        if (producer) {
            ggml_tensor * wz = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2560, ne[0]*ne[1]);
            ggml_tensor * xs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, ne[2]);
            ggml_set_param(wz); ggml_set_name(wz, "wz");
            ggml_set_param(xs); ggml_set_name(xs, "xs");
            z = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, wz, xs), ne[0], ne[1], ne[2], ne[3]);
        } else {
            z = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
            ggml_set_param(z); ggml_set_name(z, "z");
        }
        ggml_tensor * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
        ggml_tensor * out = ggml_mul(ctx, normalized, ggml_sigmoid(ctx, z));
        ggml_set_name(out, "out");
        return out;
    }
};

static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
'''),
    ],
}


def main():
    texts = {p: io.open(p, encoding="utf-8").read() for p in EDITS}
    changed = 0
    for p, pairs in EDITS.items():
        for n, (old, new) in enumerate(pairs):
            if new in texts[p]:
                continue
            if texts[p].count(old) != 1:
                sys.exit("hunk %d: anchor found %d times, expected once, in %s" % (n + 1, texts[p].count(old), p))
            texts[p] = texts[p].replace(old, new, 1)
            changed += 1
    if not changed:
        print("already applied")
        return
    for p, t in texts.items():
        io.open(p, "w", encoding="utf-8", newline="").write(t)
    print("patched %d hunk(s) in %s" % (changed, ", ".join(os.path.basename(p) for p in EDITS)))


if __name__ == "__main__":
    main()
