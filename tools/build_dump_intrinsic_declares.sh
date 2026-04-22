#!/bin/bash
# Build dump_intrinsic_declares from source.
#
# Usage:
#   bash tools/build_dump_intrinsic_declares.sh
#
# Requires LLVM build tree at LLVM_ROOT (default: /home/amax/yangz/Env/llvm-project)

LLVM_ROOT="${LLVM_ROOT:-/home/amax/yangz/Env/llvm-project}"
LLVM_CONFIG="${LLVM_ROOT}/build/bin/llvm-config"
CXX="${LLVM_ROOT}/build/bin/clang++"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/dump_intrinsic_declares.cpp"
OUT="${SCRIPT_DIR}/dump_intrinsic_declares"

echo "LLVM root: ${LLVM_ROOT}"
echo "Source:    ${SRC}"
echo "Output:    ${OUT}"

set -ex
${CXX} -o "${OUT}" "${SRC}" \
  $(${LLVM_CONFIG} --cxxflags --ldflags --libs core support) \
  -lz -lpthread -ldl -lrt -lm -fno-rtti

echo "Built: ${OUT}"
echo ""
echo "Run:"
echo "  LD_LIBRARY_PATH=${LLVM_ROOT}/build/lib:\$LD_LIBRARY_PATH ${OUT} > kb/intrinsic_declares.raw.txt"
