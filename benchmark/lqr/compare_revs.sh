#!/usr/bin/env bash
# Benchmark two revisions of StateSpaceDynamics on the same LQR workloads, and
# compare their fitted parameters against the rounding noise floor.
#
#   benchmark/lqr/compare_revs.sh BASE_REV [NEW_REV|WORKTREE] -- [run.jl options]
#
#   BASE_REV   any git revision (main, a SHA, HEAD~3); checked out as a worktree
#   NEW_REV    another revision, or omitted / "WORKTREE" for the current checkout
#              including uncommitted changes
#
# Everything after `--` goes to run.jl for both revisions (the harness itself is
# always the current checkout's, so both sides run the identical workload).
# Results land in benchmark/lqr/results/<timestamp>/: a CSV with both timings,
# a dump per revision and workload, and compare.jl's report.
#
# Example (4 threads, medium tier, 3 EM iterations, 2 reps):
# NOISE=k (default 3) perturbed refits of the base set the noise floor; NOISE=0 skips.
#
#   JULIA_NUM_THREADS=4 benchmark/lqr/compare_revs.sh main -- \
#       --workload=plqr,slqr --tier=medium --iters=3 --reps=2
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
base_rev="${1:?usage: compare_revs.sh BASE_REV [NEW_REV|WORKTREE] -- [run.jl options]}"
shift
new_rev="WORKTREE"
if [[ $# -gt 0 && "$1" != "--" ]]; then new_rev="$1"; shift; fi
[[ $# -gt 0 && "$1" == "--" ]] && shift
threads="${JULIA_NUM_THREADS:-auto}"

stamp="$(date +%Y%m%d-%H%M%S)"
out="$here/results/$stamp"
mkdir -p "$out"
work="$repo/.bench-worktrees"
mkdir -p "$work"

# An environment identical to benchmark/lqr's, but pointing at `src_dir`.
make_env() {
    local name="$1" src_dir="$2" env="$work/env-$1"
    mkdir -p "$env"
    sed "s#^StateSpaceDynamics = {path = .*#StateSpaceDynamics = {path = \"$src_dir\"}#" \
        "$here/Project.toml" > "$env/Project.toml"
    julia --project="$env" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' >/dev/null
    echo "$env"
}

checkout() {
    local rev="$1"
    if [[ "$rev" == "WORKTREE" ]]; then echo "$repo"; return; fi
    local sha dir
    sha="$(git -C "$repo" rev-parse --short "$rev")"
    dir="$work/$sha"
    [[ -d "$dir" ]] || git -C "$repo" worktree add --detach "$dir" "$sha" >/dev/null
    echo "$dir"
}

base_dir="$(checkout "$base_rev")"
new_dir="$(checkout "$new_rev")"
base_env="$(make_env base "$base_dir")"
new_env="$(make_env new "$new_dir")"

for side in base new; do
    env_var="${side}_env"
    echo "=== $side ($([[ $side == base ]] && echo "$base_rev" || echo "$new_rev"))"
    julia --project="${!env_var}" -t "$threads" "$here/run.jl" "$@" \
        --label="$side" --csv="$out/timings.csv" --dump="$out/${side}_{workload}.jls" \
        | tee "$out/${side}.log"
done

# The noise floor: the base revision again, with its initial loadings scaled by
# (1 + 1e-14). A fit that amplifies rounding (flat directions, a line search that
# stops on a different iterate) moves under that as much as under any reordering
# of floating-point work, so `new` is only distinguishable from `base` where it
# differs by more than the noise runs do among themselves and from `base`.
nnoise="${NOISE:-3}"
for k in $(seq 1 "$nnoise"); do
    delta="$(awk -v k="$k" 'BEGIN { printf "%.0e", (k % 2 ? 1 : -1) * int((k + 1) / 2) * 1e-14 }')"
    echo "=== noise floor $k/$nnoise (base, loadings × (1 + $delta))"
    julia --project="$base_env" -t "$threads" "$here/run.jl" "$@" --reps=1 \
        --perturb="$delta" --label="noise$k" --csv="$out/timings.csv" \
        --dump="$out/noise${k}_{workload}.jls" | tee "$out/noise$k.log"
done

echo "=== parameter comparison (see compare.jl for how to read it)"
for f in "$out"/base_*.jls; do
    wl="$(basename "$f" .jls)"; wl="${wl#base_}"
    noise=( "$out"/noise*_"${wl}".jls )
    [[ -e "${noise[0]}" ]] || noise=()
    echo "--- $wl" | tee -a "$out/compare.txt"
    julia --project="$here" "$here/compare.jl" "$f" "$out/new_${wl}.jls" "${noise[@]}" \
        | tee -a "$out/compare.txt" || true
done
echo "results in $out"
