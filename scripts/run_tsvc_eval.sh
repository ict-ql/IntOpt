#!/bin/bash
# End-to-end TSVC evaluation with IntOpt
#
# Usage:
#   bash scripts/run_tsvc_eval.sh [config_file]
#
# Steps:
#   1. Split TSVC into individual .ll files (if not done)
#   2. Run IntOpt batch optimization
#   3. Compile optimized IR → binaries, run them
#   4. Compare time + checksum against O3 baseline

set -e

CONFIG=${1:-config/config.yaml}
TSVC_SRC="test/TSVC_2/src"
SPLIT_DIR="test/TSVC_2/split"
OPT_DIR="test/TSVC_2/intoptplus_results"
CLANG="/home/amax/yangz/Env/llvm-project/build/bin/clang"
LLC="/home/amax/yangz/Env/llvm-project/build/bin/llc"
RESULT_DIR="$OPT_DIR/tsvc_eval"

mkdir -p "$RESULT_DIR"

# ── Step 1: Split TSVC ──
if [ ! -d "$SPLIT_DIR/ll" ] || [ -z "$(ls $SPLIT_DIR/ll/*.ll 2>/dev/null)" ]; then
    echo "=== Step 1: Split TSVC ==="
    python scripts/split_tsvc.py --tsvc-dir "$TSVC_SRC" --output-dir "$SPLIT_DIR" --compile --mem2reg
else
    echo "=== Step 1: TSVC already split, skipping ==="
fi

# ── Step 2: IntOpt batch optimization ──
echo ""
echo "=== Step 2: IntOpt batch optimization ==="
python src/main.py --mode batch \
    --input "$SPLIT_DIR/ll" \
    --output "$OPT_DIR" \
    --config "$CONFIG"

# ── Step 3: Run O3 baseline ──
echo ""
echo "=== Step 3: Running O3 baseline ==="
echo -e "name\ttime\tchecksum" > "$RESULT_DIR/baseline_o3.tsv"

for bin in "$SPLIT_DIR"/bin/*; do
    [ -x "$bin" ] || continue
    name=$(basename "$bin")
    out=$("$bin" 2>/dev/null | tail -1) || out="$name\tERROR\tERROR"
    echo "$out" >> "$RESULT_DIR/baseline_o3.tsv"
done
echo "Baseline results: $RESULT_DIR/baseline_o3.tsv"

# ── Step 4: Compile optimized IR → binaries, run ──
echo ""
echo "=== Step 4: Compile & run IntOpt optimized ==="
OPT_BIN_DIR="$RESULT_DIR/opt_bins"
mkdir -p "$OPT_BIN_DIR"

echo -e "name\ttime\tchecksum" > "$RESULT_DIR/intopt.tsv"

ok=0; fail=0; skip=0
for opt_ll in "$OPT_DIR"/*.optimized.ll; do
    [ -f "$opt_ll" ] || continue
    name=$(basename "$opt_ll" .optimized.ll)

    # Compile: optimized .ll → .s → link with main + common + dummy + globals
    asm="$OPT_BIN_DIR/${name}.s"
    bin="$OPT_BIN_DIR/${name}"
    main_c="$SPLIT_DIR/c/${name}_main.c"

    # llc: .ll → .s (with PIC for PIE linking)
    if ! $LLC -O3 --relocation-model=pic "$opt_ll" -o "$asm" 2>/dev/null; then
        echo "  $name: llc FAIL"
        fail=$((fail+1))
        echo -e "$name\tCOMPILE_FAIL\t0" >> "$RESULT_DIR/intopt.tsv"
        continue
    fi

    # clang: link .s + main + common + dummy + globals → binary
    if ! $CLANG -O3 -Wno-everything \
        -I "$TSVC_SRC" \
        "$asm" "$main_c" \
        "$TSVC_SRC/common.c" "$TSVC_SRC/dummy.c" "$SPLIT_DIR/c/globals.c" \
        -lm -o "$bin" 2>/dev/null; then
        echo "  $name: link FAIL"
        fail=$((fail+1))
        echo -e "$name\tLINK_FAIL\t0" >> "$RESULT_DIR/intopt.tsv"
        continue
    fi

    # Run (with 200s timeout to catch infinite loops)
    out=$(timeout 200 "$bin" 2>/dev/null | tail -1) || out="$name\tRUN_FAIL\t0"
    echo "$out" >> "$RESULT_DIR/intopt.tsv"
    ok=$((ok+1))
done

echo "Compiled & ran: ok=$ok  fail=$fail"
echo "IntOpt results: $RESULT_DIR/intopt.tsv"

# ── Step 5: Compare ──
echo ""
echo "=== Step 5: Compare ==="
python3 scripts/compare_tsvc.py \
    --baseline "$RESULT_DIR/baseline_o3.tsv" \
    --intopt "$RESULT_DIR/intopt.tsv" \
    --output "$RESULT_DIR/comparison.csv"

echo ""
echo "Done. Results in $RESULT_DIR/"
