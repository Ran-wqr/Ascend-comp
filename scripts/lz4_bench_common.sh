#!/usr/bin/env bash
# shellcheck shell=bash
#
# Shared dataset presets + helpers for the LZ4 benchmark drivers.
#
# >>> How to add your own dataset <<<
#   1. put the file(s) under data/ (or anywhere inside the repo)
#   2. add one line to LZ4_DATASETS below, e.g.
#        [mydata]="data/mydata/*.bin"
#   3. run any driver with that preset, e.g.
#        bash scripts/bench_cpu_vs_lz4.sh mydata 10 3
#   Or skip the table entirely and pass a glob:
#        PATTERN="data/mydata/*.bin" bash scripts/bench_cpu_vs_lz4.sh
#
# Verification: patterns under data/fp or data/large are hash-checked against
# their manifest before the run (same bytes in every session).

declare -A LZ4_DATASETS=(
  # ---- byte-stream file sets -------------------------------------------------
  [vec12]="data/t_*.bin"                 # 12 vectors, KB..2MiB
  [large1g]="data/large/*_1G.bin"        # 5 x 1 GB
  [large4g]="data/large/large_4G.bin"    # 1 x 4 GB
  [large5g]="data/large/large_5G_*.bin"  # 3 x 5 GB
  [largeall]="data/large/*.bin"          # 9 files, 24 GB
  # ---- floating-point file sets ---------------------------------------------
  [fp1m]="data/fp/*/*_1MiB.bin"          # 25 files, 1 MiB each
  [fp8m]="data/fp/*/*_8MiB.bin"          # 25 files, 8 MiB each
  [fpall]="data/fp/*/*.bin"              # 50 files, 225 MiB
  [fp16_8m]="data/fp/fp16/*_8MiB.bin"
  [bf16_8m]="data/fp/bf16/*_8MiB.bin"
  [fp32_8m]="data/fp/fp32/*_8MiB.bin"
)

lz4_presets() {
    printf '%s\n' "${!LZ4_DATASETS[@]}" | sort
}

lz4_print_presets() {
    printf '  %-10s %s\n' "preset" "glob"
    for k in $(lz4_presets); do
        printf '  %-10s %s\n' "$k" "${LZ4_DATASETS[$k]}"
    done
}

# lz4_pattern <preset> [explicit_pattern]
lz4_pattern() {
    local preset="$1" explicit="${2:-}"
    if [ -n "$explicit" ]; then printf '%s' "$explicit"; return 0; fi
    if [ -n "${PATTERN:-}" ]; then printf '%s' "$PATTERN"; return 0; fi
    if [ -z "${LZ4_DATASETS[$preset]+x}" ]; then
        printf '[dataset] unknown preset: %s\n' "$preset" >&2
        printf '[dataset] available presets:\n' >&2
        lz4_print_presets >&2
        return 2
    fi
    printf '%s' "${LZ4_DATASETS[$preset]}"
}

lz4_count_files() {
    local pat="$1" n=0 f
    for f in $pat; do [ -e "$f" ] && n=$((n + 1)); done
    printf '%s' "$n"
}

lz4_require_files() {
    local pat="$1" n
    n=$(lz4_count_files "$pat")
    if [ "$n" -eq 0 ]; then
        printf '[dataset] no files match: %s\n' "$pat" >&2
        return 3
    fi
    return 0
}

# hash-check the fixed sets; other patterns are assumed reproducible
lz4_verify_dataset() {
    local pat="$1"
    case "$pat" in
        *data/fp/*)
            printf '[dataset] verifying FP manifest ...\n'
            python3 scripts/gen_fp_data.py --verify || return 1 ;;
        *data/large/*)
            printf '[dataset] verifying large manifest ...\n'
            python3 scripts/gen_large_testdata.py --verify || return 1 ;;
    esac
    return 0
}

# dtype from a FP path: data/fp/<dtype>/...
lz4_dtype_of() {
    case "$1" in
        */fp16/*) printf 'fp16' ;;
        */bf16/*) printf 'bf16' ;;
        */fp32/*) printf 'fp32' ;;
        *)        printf '' ;;
    esac
}

lz4_banner() {  # <title> <pattern> <iters> <warmup> [extra]
    printf '=== %s ===\n' "$1"
    printf 'dataset : %s  (%s files)\n' "$2" "$(lz4_count_files "$2")"
    printf 'iters=%s warmup=%s%s\n' "$3" "$4" "${5:+  |  $5}"
}

lz4_result() {  # <rc> <label>
    if [ "$1" -eq 0 ]; then
        printf 'RESULT: PASS  (%s)\n' "$2"
        return 0
    fi
    printf 'RESULT: FAIL  (%s)\n' "$2"
    return 1
}
