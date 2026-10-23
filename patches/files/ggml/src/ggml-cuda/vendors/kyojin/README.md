# Vendored: exllamav3 / Kyojin HIP pieces (MIT)

From Kyojin (github.com/Yamz-Labs/kyojin, main of 2026-10-21), the gfx11.5 port of ExLlamaV3 (turboderp, MIT); license in
`LICENSE`. Used only by `../../mmh-moe.cu` (the grouped EXL3 MoE prefill), compiled for HIP.

- `util.h`, `util.cuh`, `compat.cuh`, `compat_rocm.cuh`, `quant/codebook.cuh`, `quant/exl3_dq.cuh`,
  `quant/hadamard_inner.cuh`, `hip/hip_mma.cuh`: as published, except `util.cuh` includes `cublas_stub.h` (ours) in place
  of `<cublas_v2.h>`.
- `mpw2.cuh`: the metadata kernel, the 128-wide Hadamard helper and the `mpw2` grouped GEMM of
  `exllamav3_ext/quant/exl3_moe_prefill.cu`; changed: the trellis tile address takes this fork's layout of EXL3 weights
  (128-row groups, see the comment there), and only the instances this fork launches are kept.
