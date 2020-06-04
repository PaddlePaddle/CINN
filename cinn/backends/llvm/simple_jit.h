#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
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
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/lang/module.h"

namespace cinn {
namespace backends {

class SimpleJIT {
 public:
  static std::unique_ptr<SimpleJIT> Create() { return std::unique_ptr<SimpleJIT>(new SimpleJIT); }

  void Link(lang::Module module, bool optimize = true);

  void Link(llvm::orc::ThreadSafeModule m, bool optimize = true) { llvm::cantFail(jit_->addIRModule(std::move(m))); }

  llvm::JITTargetAddress Lookup(const std::string& name) { return llvm::cantFail(jit_->lookup(name)).getAddress(); }

 private:
  void AddModule(std::unique_ptr<llvm::Module> module, bool optimize);

  llvm::LLVMContext& context() { return *context_.getContext(); }

  SimpleJIT() : context_(std::make_unique<llvm::LLVMContext>()) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    jit_ = llvm::cantFail(llvm::orc::LLJITBuilder().create());
    CHECK(jit_) << "JIT create failed";

    auto proc_symbols_generator = llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit_->getDataLayout().getGlobalPrefix()));
    jit_->getMainJITDylib().addGenerator(std::move(proc_symbols_generator));

    llvm::orc::MangleAndInterner mangle(jit_->getExecutionSession(), jit_->getDataLayout());

    for (auto& item : RuntimeSymbolRegistry::Global().All()) {
      LOG(INFO) << "Insert [" << item.first << "] to SimpleJIT";
      llvm::cantFail(jit_->defineAbsolute(*mangle(item.first), {llvm::pointerToJITTargetAddress(item.second), {}}));
    }
  }

  std::unique_ptr<llvm::orc::LLJIT> jit_;
  llvm::orc::ThreadSafeContext context_;
};

}  // namespace backends
}  // namespace cinn
