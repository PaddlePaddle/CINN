#pragma once

#include <llvm/IR/IRBuilder.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/backends/llvm/codegen_llvm.h"

namespace cinn::backends {

class CodeGenX86 : public CodeGenLLVM {
 public:
  explicit CodeGenX86(llvm::Module *m, llvm::IRBuilder<> *b, const std::shared_ptr<SymbolTable> &vars = nullptr);
  virtual ~CodeGenX86();

  using LLVMIRVisitor::Visit;
};

}  // namespace cinn::backends
