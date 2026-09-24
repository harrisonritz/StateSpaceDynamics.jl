#!/usr/bin/env bash
# Thread scaling of the inverse-LQR EM fit on one machine.
#
#   benchmark/lqr/scaling.sh "1 2 4 8 16 32" [run.jl options]
#
# Runs run.jl once per thread count (BLAS pinned to one thread, so Julia tasks
# are what is measured), appending to one CSV, then prints speedup and parallel
# efficiency per workload. Example, on a 32-core node:
#
#   benchmark/lqr/scaling.sh "1 4 8 16 32" --workload=plqr,slqr --tier=smoulder --iters=3
#
# Use --label=<text> to tag a run (e.g. the revision) and pass the same CSV to
# summarize.jl to compare labels side by side.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
threads_list="${1:?usage: scaling.sh \"1 2 4 ...\" [run.jl options]}"
shift
out="$here/results/scaling-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
for t in $threads_list; do
    echo "=== $t threads"
    julia --project="$here" -t "$t" "$here/run.jl" "$@" --csv="$out/timings.csv" \
        | tee "$out/threads-$t.log"
done
julia --project="$here" "$here/summarize.jl" "$out/timings.csv" | tee "$out/summary.txt"
echo "results in $out"
