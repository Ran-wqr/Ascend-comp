#!/usr/bin/env bash
# Compare current LZ4 compressed output against a stored baseline (regression guard).
#   check_baseline.sh save   -> record hashes into logs/baseline_sha256.txt
#   check_baseline.sh verify -> re-compress and compare
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
BASE=logs/baseline_sha256.txt
MODE="${1:-verify}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

fail=0
gen() {
    for f in data/t_small.bin data/t_repeat.bin data/t_text.bin data/t_struct.bin data/t_random.bin; do
        [ -s "$f" ] || continue
        for d in 1 2 4; do
            out="$TMP/$(basename "$f").d$d.lz4blocks"
            ./build/lz4_demo "$f" "$out" "$d" > /dev/null 2>&1 || { echo "[baseline] compress failed: $f d$d"; fail=1; continue; }
            echo "$(sha256sum "$out" | cut -d' ' -f1)  $(basename "$f") d$d"
        done
    done
}

if [ "$MODE" = save ]; then
    gen > "$BASE"
    echo "[baseline] saved $(wc -l < "$BASE") entries -> $BASE"
    exit 0
fi

[ -f "$BASE" ] || { echo "[baseline] missing $BASE (run: check_baseline.sh save)"; exit 2; }
gen > "$TMP/now.txt"
if diff -u "$BASE" "$TMP/now.txt" > "$TMP/diff.txt"; then
    echo "[baseline] OK: $(wc -l < "$BASE") compressed outputs unchanged"
    exit 0
else
    echo "[baseline] MISMATCH (encoding changed):"
    sed 's/^/  /' "$TMP/diff.txt" | head -40
    exit 1
fi
