#include "cinn/host_context/op_executable.h"
#include <string>
#include "cinn/host_context/kernel_frame.h"
#include "cinn/host_context/kernel_registry.h"
#include "cinn/host_context/symbol_table.h"

namespace cinn::host_context {

struct OpExecutable::Impl {
  Impl(std::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry)
      : op_name(op_name),
        symbol_table(symbol_table),
        kernel_registry(kernel_registry ? kernel_registry : GetCpuKernelRegistry()) {}

  std::string_view op_name;
  SymbolTable* symbol_table{};
  KernelFrameBuilder frame;
  KernelRegistry* kernel_registry{};

  KernelImplementation kernel_impl{};
};

OpExecutable::OpExecutable(std::string_view op_name, SymbolTable* symbol_table, KernelRegistry* kernel_registry)
    : impl_(new Impl(op_name, symbol_table, kernel_registry)) {
  // Cpu kernel registry is the default KernelRegistry.
  impl_->kernel_impl = impl_->kernel_registry->GetKernel(op_name);
  // TODO(Superjomn) support other device other than CPU.
  CHECK(impl_->kernel_impl) << "No CPU kernel called " << op_name;
}

void OpExecutable::AppendArgument(std::string_view name) {
  if (!impl_->symbol_table->Get(name)) {
    impl_->symbol_table->Register(name);
  }
  impl_->frame.AddArgument(ValueRef(impl_->symbol_table->Get(name)));
}

void OpExecutable::SetResultNum(int num) { impl_->frame.SetNumResults(num); }

KernelFrame& OpExecutable::frame() { return impl_->frame; }
const KernelFrame& OpExecutable::frame() const { return impl_->frame; }

void OpExecutable::SetResults(llvm::ArrayRef<std::string> result_names) {
  impl_->frame.SetNumResults(result_names.size());
  for (int result_id = 0; result_id < result_names.size(); result_id++) {
    Value* value = impl_->symbol_table->Register(result_names[result_id]);
    impl_->frame.SetResultAt(result_id, value);
  }
}

void OpExecutable::Execute() { impl_->kernel_impl(&impl_->frame); }

OpExecutable::~OpExecutable() {}

}  // namespace cinn::host_context
