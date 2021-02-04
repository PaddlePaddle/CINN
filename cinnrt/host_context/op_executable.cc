#include "cinnrt/host_context/op_executable.h"

#include <string>

#include "cinnrt/host_context/kernel_frame.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_function_executable.h"
#include "cinnrt/host_context/symbol_table.h"

namespace cinnrt::host_context {

struct OpExecutable::Impl {
  Impl(std::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry)
      : name(op_name),
        symbol_table(symbol_table),
        kernel_registry(kernel_registry ? kernel_registry : GetCpuKernelRegistry()) {}

  std::string name;
  SymbolTable* symbol_table{};
  KernelFrameBuilder frame;
  KernelRegistry* kernel_registry{};

  std::unique_ptr<MlirFunctionExecutable> mlir_function_executable;

  KernelImplementation kernel_impl{};
};

OpExecutable::OpExecutable(OpExecutable::Impl* impl) : impl_(impl) {}

std::string_view OpExecutable::name() const { return impl_->name; }

OpExecutableBuilder::OpExecutableBuilder(std::string_view op_name,
                                         SymbolTable* symbol_table,
                                         KernelRegistry* kernel_registry)
    : OpExecutable(new Impl(op_name, symbol_table, kernel_registry)) {
  // Cpu kernel registry is the default KernelRegistry.
  impl_->kernel_impl = impl_->kernel_registry->GetKernel(std::string(op_name.data(), op_name.size()));
  // TODO(Superjomn) support other device other than CPU.
  CHECK(impl_->kernel_impl) << "No CPU kernel called " << op_name;
}

void OpExecutableBuilder::AppendArgument(std::string_view name) {
  if (!impl_->symbol_table->GetValue(name)) {
    impl_->symbol_table->Register(name);
  }
  impl_->frame.AddArgument(impl_->symbol_table->GetValue(name));
}

void OpExecutableBuilder::AppendArgument(Value* value) { impl_->frame.AddArgument(value); }

KernelFrame& OpExecutable::frame() { return impl_->frame; }
const KernelFrame& OpExecutable::frame() const { return impl_->frame; }

void OpExecutableBuilder::SetResults(llvm::ArrayRef<std::string> result_names) {
  llvm::SmallVector<Value*, 3> results;
  for (int result_id = 0; result_id < result_names.size(); result_id++) {
    Value* value = impl_->symbol_table->Register(result_names[result_id]);
    results.push_back(value);
  }
  impl_->frame.SetResults(results);
}

void OpExecutableBuilder::SetResults(llvm::ArrayRef<Value*> results) { impl_->frame.SetResults(results); }

void OpExecutableBuilder::AppendAttribute(Value* value) { impl_->frame.AddAttribute(value); }

OpExecutableBuilder::OpExecutableBuilder(OpExecutableBuilder&& other) : OpExecutable(other.impl_.release()) {}

CoreRuntimeBuilder* OpExecutableBuilder::GetCallRuntimeBuilder() {}

MlirFunctionExecutable* OpExecutableBuilder::CreateFunctionExecutable(
    mlir::FuncOp op, MlirToRuntimeTranslator::function_defs_t* function_defs) {
  CHECK(!impl_->mlir_function_executable);
  impl_->mlir_function_executable.reset(new MlirFunctionExecutable(op, impl_->kernel_registry, *function_defs));
  return impl_->mlir_function_executable.get();
}

void OpExecutable::Execute() {
#ifndef NDEBUG
  VLOG(3) << "execute " << name() << " --- frame args: " << impl_->frame.GetNumArgs() << " results "
          << impl_->frame.GetNumResults() << " attributes " << impl_->frame.GetNumAttributes();
  for (int i = 0; i < impl_->frame.GetNumArgs(); i++) {
    VLOG(3) << "function arg: " << impl_->frame.GetArgAt(i);
  }
  for (int i = 0; i < impl_->frame.GetNumResults(); i++) {
    VLOG(3) << "function result: " << impl_->frame.GetResults()[i];
  }
#endif

  impl_->kernel_impl(&impl_->frame);
}

OpExecutable::~OpExecutable() {}

}  // namespace cinnrt::host_context
