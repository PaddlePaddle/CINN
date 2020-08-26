#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
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

#include "cinn/backends/llvm/codegen_x86.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/lang/module.h"

namespace cinn::backends {

class NaiveObjectCache : public llvm::ObjectCache {
 public:
  void notifyObjectCompiled(const llvm::Module *, llvm::MemoryBufferRef) override;
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *) override;

 private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cached_objects_;
};

struct ExecutionOptions {
  int opt_level{3};
  bool enable_debug_info{false};
  // TODO(fc500110)
  // int num_compile_threads{1};
  // bool enable_fast_math;
};

class ExecutionEngine {
 public:
  static std::unique_ptr<ExecutionEngine> Create(const ExecutionOptions &config);

  void *Lookup(std::string_view name);

  template <typename CodeGenT = CodeGenLLVM>
  void Link(const lang::Module &module);

  bool AddModule(std::unique_ptr<llvm::Module> module, std::unique_ptr<llvm::LLVMContext> context);

 protected:
  explicit ExecutionEngine(bool enable_object_cache) : cache_(std::make_unique<NaiveObjectCache>()) {}

  void RegisterRuntimeSymbols();

  bool SetupTargetTriple(llvm::Module *module);

  friend std::unique_ptr<ExecutionEngine> std::make_unique<ExecutionEngine>(bool &&);

 private:
  mutable std::mutex mu_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::unique_ptr<NaiveObjectCache> cache_;
};

}  // namespace cinn::backends
