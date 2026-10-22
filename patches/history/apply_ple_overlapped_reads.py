#!/usr/bin/env python3
"""Keep several PLE row reads in flight per worker on Windows.

llama_lazy_reader::run_range issued one read_at per row and read_at waits on its own completion
(GetOverlappedResult(..., TRUE)), so every worker held exactly one outstanding I/O. Measured with
scripts/hip-bench.ps1 -Trace on the 96 GB carve:

  prefill  262144 rows (16384 tokens x 16 heads), 64 workers, 3084-3818 ms
           = ~750 us per row per worker against a 172 us disk latency (tools/ple_io_depth.py also
             saw a queue depth of ~6 from 64 workers)
  decode   16 rows, and n_workers = min(n_threads, max(1, n/32)) rounds that to ONE worker,
           so 16 strictly sequential 90-byte reads: 2.0-6.8 ms, median 3.0 ms, every token

With depth 16 those become 2284-2376 ms and a 0.8 ms median. End to end on the production tree,
llama-bench pp16384/tg128, 3 reps: 875.30 +/- 5.25 / 23.30 +/- 1.70, against 810.82 / 23.11 for
the depth-1 reader.

Windows only - the POSIX branch keeps its pread loop untouched. Also adds gather timing under
STRIX_PLE_TRACE=1, which is how the numbers above were obtained.

Usage: python patches/apply_ple_overlapped_reads.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONOR = os.path.join(ROOT, "patches", "lazy-reader", "llama-lazy-reader.h")
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
TARGET = os.path.join(tree, "src", "llama-lazy-reader.h")


def main():
    if not os.path.isfile(TARGET):
        sys.exit("missing %s" % TARGET)
    if not os.path.isfile(DONOR):
        sys.exit("missing donor %s" % DONOR)

    donor = io.open(DONOR, encoding="utf-8").read()
    if "run_range_overlapped" not in donor:
        sys.exit("the donor reader does not carry the overlapped gather; nothing to apply")

    cur = io.open(TARGET, encoding="utf-8").read()
    if "run_range_overlapped" in cur:
        print("already applied")
        return
    io.open(TARGET, "w", encoding="utf-8", newline="").write(donor)
    print("installed the overlapped reader into", TARGET)


if __name__ == "__main__":
    main()
