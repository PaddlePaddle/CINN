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
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"

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

inline llvm::StringRef AsStringRef(absl::string_view str) { return llvm::StringRef(str.data(), str.size()); }

llvm::Type *CinnTypeToLLVMType(common::Type t, llvm::Module *m, bool is_vec = false);

template <typename T>
llvm::Type *llvm_type_of(llvm::Module *m);

}  // namespace backends
}  // namespace cinn
