#!/bin/bash
# Generate kb/intrinsic_declares.json from the dump_intrinsic_declares tool.
#
# Prerequisites:
#   bash tools/build_dump_intrinsic_declares.sh
#
# Usage:
#   bash tools/gen_intrinsic_declares_json.sh

LLVM_ROOT="${LLVM_ROOT:-/home/amax/yangz/Env/llvm-project}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL="${SCRIPT_DIR}/dump_intrinsic_declares"
RAW="${PROJECT_ROOT}/kb/intrinsic_declares.raw.txt"
JSON="${PROJECT_ROOT}/kb/intrinsic_declares.json"

if [ ! -x "${TOOL}" ]; then
  echo "Error: ${TOOL} not found. Run 'bash tools/build_dump_intrinsic_declares.sh' first."
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/kb"

echo "Step 1: Dumping intrinsic declares ..."
LD_LIBRARY_PATH="${LLVM_ROOT}/build/lib:${LD_LIBRARY_PATH}" "${TOOL}" > "${RAW}"
echo "  Raw output: ${RAW} ($(wc -l < "${RAW}") lines)"

echo "Step 2: Converting to JSON ..."
python3 -c "
import json, re
from collections import defaultdict

d = {}
for line in open('${RAW}'):
    if '|||' not in line:
        continue
    parts = line.strip().split('|||', 1)
    base_name = parts[0].strip()
    decl = parts[1].strip()
    if decl == 'OVERLOADED':
        d[base_name] = 'OVERLOADED'
    else:
        m = re.search(r'@(llvm\.[^\s(]+)', decl)
        if m:
            full_name = m.group(1)
            d[full_name] = decl

with open('${JSON}', 'w') as f:
    json.dump(d, f, indent=2)

n_overloaded = sum(1 for v in d.values() if v == 'OVERLOADED')
n_fixed = len(d) - n_overloaded
print(f'  Saved {len(d)} entries ({n_fixed} with signatures, {n_overloaded} overloaded)')
"

echo "Done: ${JSON}"
