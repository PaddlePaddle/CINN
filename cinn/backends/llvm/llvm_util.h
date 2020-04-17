#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#include <string>

#include "cinn/common/type.h"

namespace cinn {
namespace backends {

template <typename T>
std::string DumpToString(const T &entity) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  entity.print(os);
  os.flush();
  return buffer;
  // return "\033[33m" + buffer + "\033[0m"; // Green
}

llvm::Type *CinnTypeToIrType(common::Type t, llvm::Module *m);

}  // namespace backends
}  // namespace cinn
