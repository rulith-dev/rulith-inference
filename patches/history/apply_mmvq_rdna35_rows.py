#!/usr/bin/env python3
"""Give gfx1151 its own MMVQ parameter table, and two output rows per workgroup for IQ4_NL at batch 1.

get_device_table_id() mapped RDNA3.5 onto MMVQ_PARAMETERS_RDNA2, and neither calc_nwarps nor
calc_rows_per_block has an RDNA2 branch -- both fall through to 1. So every decode-batch MUL_MAT_ID
launched one single-wave workgroup per output row: 25600 of them for the MoE down projection
(ffn_down_exps, 2560 rows x 10 experts), each reading 360 bytes of IQ4_NL. The dedicated MoE kernel
that does pick 4 rows for IQ4_NL on this arch is only reached when ncols_dst > 1, which at decode it
never is.

This adds MMVQ_PARAMETERS_RDNA3_5 so real RDNA2 is untouched, keeps nwarps at 1 (what the RDNA2
table gave), and returns 2 rows per workgroup for ncols_dst == 1 and IQ4_NL only. Every wider batch
keeps the old value, so prefill is unchanged.

Measured, per-node HIP event timing at 2.3K context, three decode graphs per configuration, with the
unchanged IQ3_S gate/up kernel as a drift control (it moves +0.20 ms between runs on its own):

  ffn_moe_down  [2560,10]   1 row  4.17 +/- 0.13 ms    2 rows  3.90 +/- 0.07 ms   -0.44 ms corrected
  ffn_moe_gate  [640,10]    1 row  5.21 +/- 0.13 ms    (control)

4 rows was also measured and is a wash (4.35 +/- 0.06, i.e. drift and nothing else). The win is
about 1% of a decoded token, which is below the +/-1.5 ms run-to-run spread of a wall-clock A/B, so
do not expect to see it there. Each output row's dot product still reduces inside one wave, so the
arithmetic is untouched: tools/decode_lab.py --gen output is byte-identical to the stock kernel.

Usage: python patches/apply_mmvq_rdna35_rows.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SRC = os.path.join(tree, "ggml", "src", "ggml-cuda", "mmvq.cu")

OLD_ENUM = '''enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;'''

NEW_ENUM = '''enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

// strixllama: rows of the output each workgroup computes on RDNA3.5 when ncols_dst == 1.
// gfx1151 shared the RDNA2 table, where both calc_nwarps and calc_rows_per_block fall through to 1,
// so a decode-batch MUL_MAT_ID launched one single-wave workgroup per output row: 25600 of them for
// the MoE down projection, each reading 360 bytes of IQ4_NL. Wider blocks reuse the activation
// vector across rows and cut the grid by this factor.
#ifndef STRIX_MMVQ_RDNA35_ROWS
#define STRIX_MMVQ_RDNA35_ROWS 2
#endif

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;'''

OLD_HOST = '''    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }'''

NEW_HOST = '''    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }'''

OLD_NWARPS = '''    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {'''

NEW_NWARPS = '''    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // unchanged from when gfx1151 shared the RDNA2 table; only rows_per_block moves, see below
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {'''

OLD_ROWS_SIG = '''static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {'''
NEW_ROWS_SIG = '''static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {'''

OLD_ROWS_TAIL = '''            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool unroll_q8 = false>'''

NEW_ROWS_TAIL = '''            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // Only the decode batch moves, and only for the quant the MoE down projection is stored in.
        // Measured per-kernel at 2.3K context: down [2560,10] 4.17 -> 3.90 ms with two rows per
        // workgroup, while the IQ3_S gate/up [640,10] went the other way, so that one keeps one row.
        // Every wider batch keeps what the RDNA2 table gave it.
        return ncols_dst == 1 && type == GGML_TYPE_IQ4_NL ? STRIX_MMVQ_RDNA35_ROWS : 1;
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool unroll_q8 = false>'''

CALLS = [
    ("constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);",
     "constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);"),
    ("const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);",
     "const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);"),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    if "MMVQ_PARAMETERS_RDNA3_5" in s:
        print("already applied")
        return
    pairs = [(OLD_ENUM, NEW_ENUM), (OLD_HOST, NEW_HOST), (OLD_NWARPS, NEW_NWARPS),
             (OLD_ROWS_SIG, NEW_ROWS_SIG), (OLD_ROWS_TAIL, NEW_ROWS_TAIL)] + CALLS
    for old, _ in pairs:
        if s.count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (s.count(old), SRC, old[:120]))
    for old, new in pairs:
        s = s.replace(old, new)
    io.open(SRC, "w", encoding="utf-8", newline="").write(s)
    print("patched", SRC)


if __name__ == "__main__":
    main()
