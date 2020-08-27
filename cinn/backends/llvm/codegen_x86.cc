#include "cinn/backends/llvm/codegen_x86.h"

#include <unordered_map>

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"

namespace cinn::backends {

CodeGenX86::CodeGenX86(llvm::Module *m, llvm::IRBuilder<> *b, const std::shared_ptr<SymbolTable> &vars)
    : CodeGenLLVM(m, b, vars) {}

CodeGenX86::~CodeGenX86() {}

}  // namespace cinn::backends
