# Vendored: Crucible EXL3 decode helpers (Apache-2.0)

From Crucible (github.com/jaedync/crucible-strix-halo, main of 2026-10-23), an EXL3 inference engine for Strix Halo;
license in `LICENSE`, notice in `NOTICE`. Used by `../../mmh-moe.cu` (the decode-sized MoE pipeline) and
`../../mul-mat-had.cu` (the one-row dense ring).

- `exl3_gemv.h`: as published (`engine/exl3/kernels/exl3_gemv.h`): the mul1 trellis decode of a lane's 8 weights
  (`LaneWindow`, `DecodeLane`, `Mul1Product`, `Mul1Pair`), `Dot2` and the stream buffer resource (`StreamRsrc`).
- `lds_stream_ring.h`: as published (`engine/gfx1151/lds_stream_ring.h`): the LDS slot handoff between loader and
  compute waves.
- `../../mul-mat-had.cu` adapts Crucible's dense decode ring (`engine/exl3/exl3_ring.h`: `LinRingFlowKernel`, `Load`,
  `Compute`, `ConsumerBarrier`): two loader waves stream 4-tile chunks of a 128-row group into a 3-slot LDS ring with
  128-bit loads, eight compute waves decode them. Changed: a chunk is read where this fork's layout keeps it (the 4 k
  rows of a 128-row group are already contiguous, so no front-order copy of the weights), the input transform and the
  output transform and store are this fork's (`mmh_fwht`, the rotations, `mmh_store`).
- `../../mul-mat-had.cu` also adapts Crucible's R-row ring (`engine/exl3/ring_rows.h`: `LinRingRowsKernel`, `Load`,
  `Compute`) for the 2 to 4 rows of MTP verification: the loader waves stage each chunk's activations (R rows x 4 tiles
  x 8 words) behind its weights in the ring slot. Changed: the rows' input is this fork's transform, precomputed by
  `mmh_ring_rows_in`; the splits, the per-row arithmetic and the epilogue are those of the one-row ring, so a row's
  result does not depend on R.
- `../../mmh-moe.cu` runs the decode-sized expert jobs by the same ring (`moe_ring::moe_dec_ring`, after
  `LinRingRowsKernel` and `MoeFusedKernel`): every job's gate and up groups, then its down groups, a down item waiting
  for its job's gate/up items. Changed: the item order and the job tables are this fork's, the down item's input is the
  SwiGLU step of `moe_act` computed into LDS, one k span per item so that the products equal the fused launch's. Ours
  on top: the workgroups take items from a counter (heaviest first) instead of a fixed round-robin, and the shared
  expert runs as one more job of the same launch (its own widths: a second loader / decoder instance).
- `../../mmh-moe.cu` adapts Crucible's fused MoE decode (`engine/layers/moe_rows.h`: `MoeGemvCols`, `MoeFusedKernel`):
  the tile loop of a wave over its column blocks and the gate/up-then-down launch in which each down workgroup waits
  for its expert's gate/up workgroups. Changed: the tile addresses follow this fork's layout of EXL3 weights (128-row
  groups, k-major inside a group), the rows of a job are its picks, and the step between gate/up and down (output
  rotations, SwiGLU, the down input rotation) is this fork's F32 arithmetic of `moe_act`.
