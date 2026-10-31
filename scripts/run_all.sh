#!/usr/bin/env bash
# Interop / regression suite.
#
# Status model:
#   PASS : the check actually ran and all assertions passed
#   FAIL : the check ran and failed (or errored)
#   SKIP : the check was not run (missing dependency / not applicable)
#
# Options:
#   --strict   treat any SKIP as a failure (exit 2)
#
# Columns:
#   A NPU round-trip   B NPU output -> official decoder
#   C official encoder -> NPU decode   D NPU decompressed-size query
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh

STRICT=0
for a in "$@"; do
    case "$a" in
        --strict) STRICT=1 ;;
        *) echo "unknown option: $a"; exit 2 ;;
    esac
done

LZ4=build/lz4_demo
SNAP=build/snappy_demo
RLZ4=ref/tools/ref_lz4
RSNAP=ref/tools/ref_snappy
LOGDIR=logs
mkdir -p "$LOGDIR" data
OUT="$LOGDIR/run_all.log"
SKIPLOG="$LOGDIR/skips.log"
FAILLOG="$LOGDIR/failures.log"
: > "$OUT"; : > "$SKIPLOG"; : > "$FAILLOG"

[ -x "$LZ4" ]  || { echo "missing $LZ4 (run scripts/build.sh)"; exit 1; }
[ -x "$SNAP" ] || { echo "missing $SNAP (run scripts/build.sh)"; exit 1; }

REF_OK=0
if [ -x "$RLZ4" ] && [ -x "$RSNAP" ]; then REF_OK=1; fi

python3 scripts/gen_testdata.py > "$LOGDIR/gen_testdata.log" 2>&1 || { echo "[run_all] gen_testdata failed"; exit 1; }
python3 scripts/gen_batch.py    > "$LOGDIR/gen_batch.log" 2>&1    || { echo "[run_all] gen_batch failed"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

N_PASS=0; N_FAIL=0; N_SKIP=0

mark() {   # $1 = PASS|FAIL|SKIP, $2 = context
    case "$1" in
        PASS) N_PASS=$((N_PASS+1)) ;;
        FAIL) N_FAIL=$((N_FAIL+1)); echo "$2" >> "$FAILLOG" ;;
        SKIP) N_SKIP=$((N_SKIP+1)); echo "$2" >> "$SKIPLOG" ;;
    esac
}

ref_reason() {
    if [ "$REF_OK" = 1 ]; then echo ""; else echo "reference codec not built (run scripts/build_ref.sh)"; fi
}

run_case () {   # $1 algo  $2 file  $3 dtype
    algo="$1"; f="$2"; dtype="${3:-1}"; base="$(basename "$f")"
    if [ "$algo" = lz4 ]; then DEMO="$LZ4"; EXT="lz4blocks"; REF="$RLZ4"; tag="lz4/d$dtype";
    else DEMO="$SNAP"; EXT="snappyblocks"; REF="$RSNAP"; tag="snappy"; fi
    blocks="$f.$EXT.$dtype"

    if [ "$algo" = lz4 ]; then
        "$DEMO" "$f" "$blocks" "$dtype" > "$TMP/a.log" 2>&1
    else
        "$DEMO" "$f" "$blocks" > "$TMP/a.log" 2>&1
    fi
    if grep -q "ROUNDTRIP OK" "$TMP/a.log"; then A=PASS; else A=FAIL; fi
    mark "$A" "A $tag $base (npu round-trip)"

    if [ "$REF_OK" = 0 ]; then
        B=SKIP; C=SKIP
        mark "$B" "B $tag $base: $(ref_reason)"
        mark "$C" "C $tag $base: $(ref_reason)"
    else
        if "$REF" decompress "$blocks" "$TMP/b.out" > "$TMP/b.log" 2>&1 && cmp -s "$f" "$TMP/b.out"; then B=PASS; else B=FAIL; fi
        mark "$B" "B $tag $base (ref decodes NPU)"
        "$REF" compress "$f" "$TMP/c.blocks" 65536 > "$TMP/c1.log" 2>&1
        if "$DEMO" --decode "$TMP/c.blocks" "$TMP/c.out" > "$TMP/c2.log" 2>&1 && cmp -s "$f" "$TMP/c.out"; then C=PASS; else C=FAIL; fi
        mark "$C" "C $tag $base (NPU decodes ref)"
    fi

    if "$DEMO" --size "$blocks" "$TMP/d.sizes" > "$TMP/d.log" 2>&1 && \
       python3 scripts/check_sizes.py "$TMP/d.sizes" "$f" 65536 > /dev/null 2>&1; then D=PASS; else D=FAIL; fi
    mark "$D" "D $tag $base (size query)"

    printf '%-14s %-24s A=%-4s B=%-4s C=%-4s D=%-4s\n' "$tag" "$base" "$A" "$B" "$C" "$D" >> "$OUT"
}

run_batch_case () {   # $1 algo  $2 dtype
    algo="$1"; dtype="${2:-1}"; LIST="data/batch/list.txt"
    if [ "$algo" = lz4 ]; then DEMO="$LZ4"; EXT="lz4blocks"; REF="$RLZ4"; tag="batch/lz4d$dtype";
        "$DEMO" --batch "$LIST" "$dtype" > "$TMP/ba.log" 2>&1
    else DEMO="$SNAP"; EXT="snappyblocks"; REF="$RSNAP"; tag="batch/snappy";
        "$DEMO" --batch "$LIST" > "$TMP/ba.log" 2>&1; fi
    if grep -q "ROUNDTRIP OK" "$TMP/ba.log"; then A=PASS; else A=FAIL; fi
    mark "$A" "A $tag batch round-trip"

    B=SKIP; C=SKIP
    if [ "$REF_OK" = 0 ]; then
        mark "$B" "B $tag: $(ref_reason)"; mark "$C" "C $tag: $(ref_reason)"
    else
        if "$REF" decompress "$LIST.$EXT" "$TMP/bb.out" > "$TMP/bb.log" 2>&1 && \
           python3 scripts/check_batch.py "$LIST" "$TMP/bb.out" > /dev/null 2>&1; then B=PASS; else B=FAIL; fi
        mark "$B" "B $tag (ref decodes NPU batch)"
        : > "$TMP/bc.blocks"
        while read -r f; do
            [ -n "$f" ] || continue; [ -s "$f" ] || continue
            "$REF" compress "$f" "$TMP/bc.item" 65536 > /dev/null 2>&1
            cat "$TMP/bc.item" >> "$TMP/bc.blocks"
        done < "$LIST"
        if "$DEMO" --decode "$TMP/bc.blocks" "$TMP/bc.out" > "$TMP/bc.log" 2>&1 && \
           python3 scripts/check_batch.py "$LIST" "$TMP/bc.out" > /dev/null 2>&1; then C=PASS; else C=FAIL; fi
        mark "$C" "C $tag (NPU decodes ref batch)"
    fi

    printf '%-14s %-24s A=%-4s B=%-4s C=%-4s D=%-4s\n' "$tag" "list(40 items)" "$A" "$B" "$C" "N/A" >> "$OUT"
}

for f in data/*.bin; do run_case lz4 "$f" 1; done
for f in data/*.bin; do run_case lz4 "$f" 2; done
for f in data/*.bin; do run_case lz4 "$f" 4; done
for f in data/*.bin; do run_case snappy "$f"; done
run_batch_case lz4 1
run_batch_case lz4 4
run_batch_case snappy

TOTAL=$((N_PASS + N_FAIL + N_SKIP))
{
echo "=================== nvCOMP LZ4 / Snappy on Ascend NPU ==================="
echo "NPU   : $(npu-smi info 2>/dev/null | grep -m1 -o 'Ascend[0-9A-Za-z]*' || echo unknown)"
echo "A = NPU round-trip | B = official decoder reads NPU output"
echo "C = NPU reads official output | D = NPU decompressed-size query"
echo "strict mode: $STRICT"
echo "------------------------------------------------------------------------"
cat "$OUT"
echo "------------------------------------------------------------------------"
echo "checks: total=$TOTAL pass=$N_PASS fail=$N_FAIL skip=$N_SKIP"
if [ -s "$SKIPLOG" ]; then echo "skipped:"; sed 's/^/  - /' "$SKIPLOG"; fi
if [ -s "$FAILLOG" ]; then echo "failed:"; sed 's/^/  - /' "$FAILLOG"; fi
if [ "$N_FAIL" -gt 0 ]; then
    echo "RESULT: FAIL"
elif [ "$STRICT" = 1 ] && [ "$N_SKIP" -gt 0 ]; then
    echo "RESULT: FAIL (strict mode: $N_SKIP skipped)"
else
    echo "RESULT: PASS ($N_SKIP skipped)"
fi
} | tee "$LOGDIR/summary.log"

if [ "$N_FAIL" -gt 0 ]; then exit 1; fi
if [ "$STRICT" = 1 ] && [ "$N_SKIP" -gt 0 ]; then exit 2; fi
exit 0
