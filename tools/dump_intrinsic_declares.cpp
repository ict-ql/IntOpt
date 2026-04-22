/**
 * dump_intrinsic_declares.cpp
 *
 * Uses LLVM's Intrinsic API to dump the correct declare signature for
 * every intrinsic. This is the authoritative source — no guessing from
 * test cases.
 *
 * Build:
 *   clang++ -o dump_intrinsic_declares dump_intrinsic_declares.cpp \
 *     $(llvm-config --cxxflags --ldflags --libs core support) \
 *     -lLLVM -std=c++17
 *
 * Or with the build tree:
 *   clang++ -o dump_intrinsic_declares dump_intrinsic_declares.cpp \
 *     -I/path/to/llvm-project/build/include \
 *     -I/path/to/llvm-project/llvm/include \
 *     -L/path/to/llvm-project/build/lib \
 *     -lLLVM -std=c++17 -fno-rtti
 *
 * Run:
 *   ./dump_intrinsic_declares > intrinsic_declares.txt
 *
 * Output format (one per line):
 *   llvm.abs ||| declare i32 @llvm.abs.i32(i32, i1 immarg)
 *   llvm.abs ||| declare <4 x i32> @llvm.abs.v4i32(<4 x i32>, i1 immarg)
 *   llvm.x86.sse2.pmovmskb.128 ||| declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>)
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

#include <string>
#include <vector>

using namespace llvm;

// Common types to instantiate overloaded intrinsics with
static std::vector<std::vector<Type *>> getCommonTypeVariants(LLVMContext &Ctx) {
  std::vector<std::vector<Type *>> variants;

  // Scalar integer types
  variants.push_back({Type::getInt8Ty(Ctx)});
  variants.push_back({Type::getInt16Ty(Ctx)});
  variants.push_back({Type::getInt32Ty(Ctx)});
  variants.push_back({Type::getInt64Ty(Ctx)});

  // Scalar float types
  variants.push_back({Type::getFloatTy(Ctx)});
  variants.push_back({Type::getDoubleTy(Ctx)});
  variants.push_back({Type::getHalfTy(Ctx)});

  // Common vector types
  variants.push_back({FixedVectorType::get(Type::getFloatTy(Ctx), 4)});
  variants.push_back({FixedVectorType::get(Type::getFloatTy(Ctx), 8)});
  variants.push_back({FixedVectorType::get(Type::getFloatTy(Ctx), 16)});
  variants.push_back({FixedVectorType::get(Type::getDoubleTy(Ctx), 2)});
  variants.push_back({FixedVectorType::get(Type::getDoubleTy(Ctx), 4)});
  variants.push_back({FixedVectorType::get(Type::getDoubleTy(Ctx), 8)});
  variants.push_back({FixedVectorType::get(Type::getInt8Ty(Ctx), 16)});
  variants.push_back({FixedVectorType::get(Type::getInt8Ty(Ctx), 32)});
  variants.push_back({FixedVectorType::get(Type::getInt8Ty(Ctx), 64)});
  variants.push_back({FixedVectorType::get(Type::getInt16Ty(Ctx), 8)});
  variants.push_back({FixedVectorType::get(Type::getInt16Ty(Ctx), 16)});
  variants.push_back({FixedVectorType::get(Type::getInt16Ty(Ctx), 32)});
  variants.push_back({FixedVectorType::get(Type::getInt32Ty(Ctx), 4)});
  variants.push_back({FixedVectorType::get(Type::getInt32Ty(Ctx), 8)});
  variants.push_back({FixedVectorType::get(Type::getInt32Ty(Ctx), 16)});
  variants.push_back({FixedVectorType::get(Type::getInt64Ty(Ctx), 2)});
  variants.push_back({FixedVectorType::get(Type::getInt64Ty(Ctx), 4)});
  variants.push_back({FixedVectorType::get(Type::getInt64Ty(Ctx), 8)});

  // Pointer type
  variants.push_back({PointerType::getUnqual(Ctx)});

  return variants;
}

static void printDeclare(Function *F, Intrinsic::ID id) {
  if (!F) return;

  // Get the base intrinsic name (without type suffix)
  std::string baseName = Intrinsic::getName(id).str();

  // Print: baseName ||| declare <full signature>
  llvm::outs() << baseName << " ||| ";

  // Print the declare line
  FunctionType *FT = F->getFunctionType();
  llvm::outs() << "declare ";
  FT->getReturnType()->print(llvm::outs());
  llvm::outs() << " @" << F->getName() << "(";

  for (unsigned i = 0; i < FT->getNumParams(); ++i) {
    if (i > 0) llvm::outs() << ", ";
    FT->getParamType(i)->print(llvm::outs());

    // Check for immarg attribute
    if (F->hasParamAttribute(i, Attribute::ImmArg))
      llvm::outs() << " immarg";
  }

  if (FT->isVarArg()) {
    if (FT->getNumParams() > 0) llvm::outs() << ", ";
    llvm::outs() << "...";
  }

  llvm::outs() << ")\n";
}

int main() {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("intrinsic_dump", Ctx);

  // Set a target triple so target-specific intrinsics work
  M->setTargetTriple("x86_64-unknown-linux-gnu");

  auto typeVariants = getCommonTypeVariants(Ctx);

  unsigned numIntrinsics = Intrinsic::num_intrinsics;

  for (unsigned id = 1; id < numIntrinsics; ++id) {
    Intrinsic::ID iid = static_cast<Intrinsic::ID>(id);

    if (!Intrinsic::isOverloaded(iid)) {
      // Non-overloaded: get the single declaration directly
      Function *F = Intrinsic::getDeclaration(M.get(), iid);
      printDeclare(F, iid);
    } else {
      // Overloaded: just record the base name so we know it exists
      std::string baseName = Intrinsic::getName(iid).str();
      llvm::outs() << baseName << " ||| OVERLOADED\n";
    }
  }

  return 0;
}
