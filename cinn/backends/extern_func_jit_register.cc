#include "cinn/backends/extern_func_jit_register.h"

namespace cinn {
namespace backends {

void RegisterExternFunctionHelper(const std::string &fn_name,
                                  std::unique_ptr<FunctionProto> &&fn_proto,
                                  Target target,
                                  void *fn_ptr) {
  ExternFunctionProtoRegistry::Global().Register(fn_name, fn_proto.release());
  CHECK(ExternFunctionProtoRegistry::Global().Lookup(fn_name));

  ExternFunctionEmitterRegistry::Global().Register(ExternFuncID{TargetToBackendRepr(target), fn_name.c_str()},
                                                   new backends::ExternFunctionLLVMEmitter(fn_name));

  RuntimeSymbolRegistry::Global().Register(fn_name, reinterpret_cast<void *>(fn_ptr));
}

void RegisterExternFunction::End() {
  auto fn_proto = fn_proto_builder_.Build();
  RegisterExternFunctionHelper(fn_name_, std::move(fn_proto), target_, fn_ptr_);
}

}  // namespace backends
}  // namespace cinn
