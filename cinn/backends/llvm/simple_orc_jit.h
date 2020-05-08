#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace backends {

class SimpleOrcJit {
 public:
  static std::unique_ptr<SimpleOrcJit> Create();

  void AddModule(std::unique_ptr<llvm::Module> module, bool optimize = false);
  void Link(const lang::Module &module, bool optimize = false);

  void *Lookup(std::string_view name);

  llvm::LLVMContext &context() { return *context_.getContext(); }

 protected:
  SimpleOrcJit(llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout data_layout);
  void RegisterRuntimeSymbols();

 private:
  SimpleOrcJit() = delete;

  mutable std::mutex mu_;
  llvm::orc::ThreadSafeContext context_;

  std::vector<llvm::orc::VModuleKey> module_keys_;

  llvm::DataLayout data_layout_;
  llvm::orc::ExecutionSession execution_session_;
  llvm::orc::RTDyldObjectLinkingLayer object_layer_;
  llvm::orc::MangleAndInterner mangle_;

  llvm::orc::IRCompileLayer compile_layer_;
  llvm::orc::JITDylib &main_jd_;
};

}  // namespace backends
}  // namespace cinn
