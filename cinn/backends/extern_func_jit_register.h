/**
 * \file This file defines some functions and macros to help register the extern functions into JIT.
 */
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/extern_func_emitter.h"
#include "cinn/backends/extern_func_emitter_builtin.h"
#include "cinn/backends/extern_func_protos.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/ir_builder_mixin.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"

#define REGISTER_EXTERN_FUNC_HELPER(fn__, target__) \
  ::cinn::backends::RegisterExternFunction(#fn__, target__, reinterpret_cast<void*>(fn__))

#define REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(fn__, target__, in_type__, out_type__) \
  REGISTER_EXTERN_FUNC_HELPER(fn__, target__).SetRetType<out_type__>().AddInputType<in_type__>().End()

#define REGISTER_EXTERN_FUNC(symbol__) bool __cinn__##symbol__##__registrar()
#define USE_EXTERN_FUNC(symbol__)                \
  extern bool __cinn__##symbol__##__registrar(); \
  [[maybe_unused]] static bool __cinn_extern_registrar_##symbol__ = __cinn__##symbol__##__registrar();

namespace cinn {
namespace backends {

static const char* TargetToBackendRepr(Target target) {
  if (target.arch == Target::Arch::X86) {
    return backend_llvm_host;
  } else {
    CINN_NOT_IMPLEMENTED
  }
  return nullptr;
}

struct RegisterExternFunction {
  RegisterExternFunction(const std::string& fn_name, Target target, void* fn_ptr)
      : fn_name_(fn_name), target_(target), fn_ptr_(fn_ptr), fn_proto_builder_(fn_name) {}

  template <typename T>
  RegisterExternFunction& AddInputType() {
    fn_proto_builder_.AddInputType<T>();
    return *this;
  }
  template <typename T>
  RegisterExternFunction& AddOutputType() {
    fn_proto_builder_.AddOutputType<T>();
    return *this;
  }
  template <typename T>
  RegisterExternFunction& SetRetType() {
    fn_proto_builder_.SetRetType<T>();
    return *this;
  }
  RegisterExternFunction& SetShapeInference(FunctionProto::shape_inference_t handle) {
    fn_proto_builder_.SetShapeInference(handle);
    return *this;
  }

  void End();

 private:
  const std::string& fn_name_;
  Target target_;
  void* fn_ptr_{};
  FunctionProto::Builder fn_proto_builder_;
};

}  // namespace backends
}  // namespace cinn
