// Monte Carlo check of speculative sampling (patches/apply_spec_sampling_042.py): tools/spec_sampling_mc.py slices
// the functions under test out of the built tree's common/sampling.cpp into spec_fns.inc (spec_draw, spec_q_of,
// common_sampler_spec_draw_q, spec_select), so this is the shipped code, compiled alone.
//
// A: one position - a token drawn from q (common_sampler_spec_draw_q, random parameters) and verified by spec_select
//    against a random p: the output must follow p, the draft q, the acceptance sum min(p, q).
// B: a stream - position-dependent Markov p and q, drafts of 1-3 tokens cut by a confidence rule on q (as p_min),
//    verified position by position like common_sampler_sample_and_accept_n_spec, a bonus token after a draft accepted
//    whole: the joint law of the first three tokens must be p's.
// C: the same stream with the n_min rule (a draft shorter than 2 dropped): expected to be biased - why the drafter
//    keeps short drawn drafts when p_min > 0.

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#undef GGML_ASSERT
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "assert: %s\n", #x); abort(); } } while (0)

#include "spec_fns.inc"

static int n_fail = 0;

static void check_dist(const char * what, const std::vector<double> & want, const std::vector<int64_t> & got, int64_t n, double z_max = 5.0) {
    double worst = 0.0;
    int    worst_i = -1;
    double chi2 = 0.0;
    int    dof = -1;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = want[i] * n;
        if (want[i] <= 0.0) {
            if (got[i] != 0) {
                printf("  %s: cell %zu has p = 0 but %lld hits\n", what, i, (long long) got[i]);
                n_fail++;
            }
            continue;
        }
        dof++;
        const double sd = std::sqrt(n * want[i] * (1.0 - want[i])) + 1e-12;
        const double z  = std::fabs(got[i] - e) / sd;
        chi2 += (got[i] - e) * (got[i] - e) / e;
        if (z > worst) {
            worst   = z;
            worst_i = (int) i;
        }
    }
    // chi2 with dof degrees of freedom: mean dof, sd sqrt(2 dof)
    const double chi2_z = dof > 0 ? (chi2 - dof) / std::sqrt(2.0 * dof) : 0.0;
    const bool ok = worst < z_max && chi2_z < z_max;
    printf("  %-34s worst cell z %.2f (cell %d), chi2 %.1f on %d dof (z %.2f) %s\n", what, worst, worst_i, chi2, dof, chi2_z, ok ? "ok" : "FAIL");
    if (!ok) {
        n_fail++;
    }
}

// A ---------------------------------------------------------------------------------------------------------------

static void test_one_position(std::mt19937 & meta, int trial) {
    const int V = 14;

    // p: a random distribution over a random subset (absent tokens have p = 0), as cur_p after the target's chain
    std::vector<llama_token_data> pc;
    {
        std::normal_distribution<double> nd(0.0, 1.6);
        std::uniform_int_distribution<int> keep(0, 3);
        double sum = 0.0;
        for (int t = 0; t < V; ++t) {
            if (keep(meta) == 0 && t > 0) {
                continue;
            }
            const double w = std::exp(nd(meta));
            pc.push_back({ t, (float) std::log(w), (float) w });
            sum += w;
        }
        for (auto & e : pc) {
            e.p = (float) (e.p / sum);
        }
        std::shuffle(pc.begin(), pc.end(), meta);
    }
    std::vector<double> want(V, 0.0);
    for (const auto & e : pc) {
        want[e.id] = e.p;
    }
    {
        // the normalization in float: renormalize want to what spec_select sees
        double s = 0.0;
        for (double x : want) s += x;
        for (double & x : want) x /= s;
    }

    // the draft's candidates: 10 random tokens, sorted by logit, some -inf (suppressed)
    std::vector<llama_token_data> cand;
    {
        std::vector<int> ids(V);
        for (int t = 0; t < V; ++t) ids[t] = t;
        std::shuffle(ids.begin(), ids.end(), meta);
        std::normal_distribution<double> nd(0.0, 2.0);
        for (int k = 0; k < 10; ++k) {
            cand.push_back({ ids[k], (float) nd(meta), 0.0f });
        }
        if (trial % 3 == 0) {
            cand[9].logit = -INFINITY;
        }
        std::sort(cand.begin(), cand.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    }
    llama_token_data_array cands = { cand.data(), cand.size(), -1, true };

    const float temps[]  = { 0.3f, 0.7f, 1.0f };
    const int   topks[]  = { 0, 3, 20 };
    const float topps[]  = { 1.0f, 0.8f, 0.5f };
    const float minps[]  = { 0.0f, 0.05f, 0.2f };
    const float scales[] = { 0.3f, 1.0f, 1.5f };
    const float temp   = temps [trial % 3];
    const int   top_k  = topks [(trial / 3) % 3];
    const float top_p  = topps [(trial / 9) % 3];
    const float min_p  = minps [(trial / 27) % 3];
    const float qscale = scales[(trial / 81) % 3];

    std::mt19937 rng_d(1000 + trial), rng_v(2000 + trial);
    std::vector<llama_token_data> q;
    std::vector<double> w;

    const int64_t N = 400000;
    std::vector<int64_t> out(V, 0), drawn(V, 0);
    int64_t n_acc = 0;

    llama_token_data_array cur_p = { pc.data(), pc.size(), -1, false };

    for (int64_t it = 0; it < N; ++it) {
        const llama_token x = common_sampler_spec_draw_q(&cands, temp, top_k, top_p, min_p, qscale, rng_d, q);
        drawn[x]++;
        bool stop = false;
        const size_t sel = spec_select(cur_p, &q, x, rng_v, w, &stop);
        const llama_token y = cur_p.data[sel].id;
        out[y]++;
        if (!stop) {
            GGML_ASSERT(y == x);
            n_acc++;
        } else {
            GGML_ASSERT(y != x || spec_q_of(q, x) <= 0.0);
        }
    }

    std::vector<double> qv(V, 0.0);
    double acc_want = 0.0;
    for (const auto & e : q) {
        qv[e.id] = e.p;
    }
    for (int t = 0; t < V; ++t) {
        acc_want += std::min(want[t], qv[t]);
    }
    const double acc = (double) n_acc / N;
    const double acc_sd = std::sqrt(acc_want * (1.0 - acc_want) / N) + 1e-12;

    printf("A%-3d temp %.1f top_k %2d top_p %.1f min_p %.2f qscale %.1f |q| %zu: acceptance %.4f, sum min(p,q) %.4f (z %.2f)\n",
            trial, temp, top_k, top_p, min_p, qscale, q.size(), acc, acc_want, std::fabs(acc - acc_want) / acc_sd);
    if (std::fabs(acc - acc_want) / acc_sd > 5.0) {
        n_fail++;
        printf("  acceptance FAIL\n");
    }
    {
        double s = 0.0;
        for (double x : qv) s += x;
        for (double & x : qv) x /= s;
    }
    check_dist("draft follows q", qv, drawn, N);
    check_dist("output follows p", want, out, N);
}

// B, C --------------------------------------------------------------------------------------------------------------

struct markov {
    static const int V = 4;
    // p[t][prev][y], q[t][prev][y] for positions t = 0.. (prev = the token before)
    std::vector<std::vector<std::vector<double>>> p, q;

    markov(int T, std::mt19937 & meta) {
        std::normal_distribution<double> nd(0.0, 1.3);
        auto rnd = [&](double sharp) {
            std::vector<double> v(V);
            double s = 0.0;
            for (int y = 0; y < V; ++y) {
                v[y] = std::exp(sharp * nd(meta));
                s += v[y];
            }
            for (auto & x : v) x /= s;
            return v;
        };
        p.resize(T);
        q.resize(T);
        for (int t = 0; t < T; ++t) {
            for (int prev = 0; prev < V; ++prev) {
                p[t].push_back(rnd(1.0));
                // q: p blended with noise, sometimes a token p never takes
                auto n = rnd(1.0);
                std::vector<double> v(V);
                double s = 0.0;
                for (int y = 0; y < V; ++y) {
                    v[y] = 0.6 * p[t][prev][y] + 0.4 * n[y];
                    s += v[y];
                }
                for (auto & x : v) x /= s;
                q[t].push_back(v);
            }
            // a zero in p somewhere
            p[t][t % V][(t + 1) % V] = 0.0;
            double s = 0.0;
            for (double x : p[t][t % V]) s += x;
            for (double & x : p[t][t % V]) x /= s;
        }
    }
};

static void stream_test(const char * name, int n_min, double conf, std::mt19937 & meta) {
    const int V = markov::V;
    const int T = 8;
    markov m(T, meta);

    std::mt19937 rng_d(7), rng_v(8);
    std::vector<double> w;

    const int64_t N = 3000000;
    std::vector<int64_t> got(V * V * V, 0);
    int64_t n_steps = 0, n_drafted = 0, n_accepted = 0;

    std::vector<llama_token_data> pc(V);

    for (int64_t it = 0; it < N; ++it) {
        std::vector<int> out;
        int prev = 0;
        while ((int) out.size() < 3) {
            const int t0 = (int) out.size();

            // draft: up to 3 tokens while q's top probability at the next position is >= 0.45 (decided before drawing)
            std::vector<int> draft;
            std::vector<std::vector<llama_token_data>> dq;
            int dprev = prev;
            for (int k = 0; k < 3 && t0 + k < T; ++k) {
                const auto & qv = m.q[t0 + k][dprev];
                if (*std::max_element(qv.begin(), qv.end()) < conf) {
                    break;
                }
                // q as a candidate list for common_sampler_spec_draw_q: logits log q, temperature 1, nothing cut
                std::vector<llama_token_data> cand;
                for (int y = 0; y < V; ++y) {
                    cand.push_back({ y, (float) std::log(qv[y]), 0.0f });
                }
                std::sort(cand.begin(), cand.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
                llama_token_data_array ca = { cand.data(), cand.size(), -1, true };
                dq.emplace_back();
                const llama_token x = common_sampler_spec_draw_q(&ca, 1.0f, 0, 1.0f, 0.0f, 1.0f, rng_d, dq.back());
                draft.push_back(x);
                dprev = x;
            }
            if ((int) draft.size() < n_min) {
                draft.clear();
                dq.clear();
            }

            // verify like common_sampler_sample_and_accept_n_spec
            n_steps++;
            n_drafted += draft.size();
            int vprev = prev;
            for (size_t i = 0; i <= draft.size(); ++i) {
                const int t = (int) out.size();
                if (t >= T) {
                    break;
                }
                const auto & pv = m.p[t][vprev];
                for (int y = 0; y < V; ++y) {
                    pc[y] = { y, 0.0f, (float) pv[y] };
                }
                llama_token_data_array cur_p = { pc.data(), pc.size(), -1, false };
                const bool has_draft = i < draft.size();
                bool stop = true;
                const size_t sel = spec_select(cur_p, has_draft ? &dq[i] : nullptr, has_draft ? draft[i] : -1, rng_v, w, &stop);
                const int y = cur_p.data[sel].id;
                out.push_back(y);
                vprev = y;
                if (stop || y != draft[i]) {
                    break;
                }
                n_accepted++;
            }
            prev = out.back();
        }
        got[(out[0] * V + out[1]) * V + out[2]]++;
    }

    std::vector<double> want(V * V * V, 0.0);
    for (int a = 0; a < V; ++a) {
        for (int b = 0; b < V; ++b) {
            for (int c = 0; c < V; ++c) {
                want[(a * V + b) * V + c] = m.p[0][0][a] * m.p[1][a][b] * m.p[2][b][c];
            }
        }
    }
    printf("%s: %lld steps, %.2f drafted a step, %.3f of drafted accepted\n", name, (long long) n_steps,
            (double) n_drafted / n_steps, n_drafted ? (double) n_accepted / n_drafted : 0.0);
    check_dist("first three tokens follow p", want, got, N);
}

int main() {
    std::mt19937 meta(12345);

    for (int trial = 0; trial < 243; trial += 7) {
        test_one_position(meta, trial);
    }

    std::mt19937 meta_b(777);
    stream_test("B stream, confidence-cut drafts", 0, 0.45, meta_b);
    std::mt19937 meta_b2(778);
    stream_test("B2 stream, drafts of 3", 0, 0.0, meta_b2);
    std::mt19937 meta_b3(779);
    stream_test("B3 stream, confidence 0.3", 0, 0.3, meta_b3);

    const int fails_before_c = n_fail;
    std::mt19937 meta_c(777);
    stream_test("C stream, drafts shorter than 2 dropped (expected biased)", 2, 0.45, meta_c);
    const bool c_biased = n_fail > fails_before_c;
    n_fail = fails_before_c;

    printf("\n%s (C %s)\n", n_fail == 0 ? "ALL OK" : "FAILURES", c_biased ? "biased, as expected" : "not detectably biased");
    return n_fail == 0 ? 0 : 1;
}
