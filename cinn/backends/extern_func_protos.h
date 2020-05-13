#pragma once

#include "cinn/backends/function_prototype.h"

namespace cinn {
namespace backends {

static const char* extern_func__tanh   = "tanh";
static const char* extern_func__tanh_v = "tanh_v";

class ExternFunctionProtoRegistry : public FunctionProtoRegistry {
 public:
  using FunctionProtoRegistry::Lookup;
  using FunctionProtoRegistry::Register;

  static ExternFunctionProtoRegistry& Global() {
    static ExternFunctionProtoRegistry x;
    return x;
  }

 private:
  ExternFunctionProtoRegistry();
  CINN_DISALLOW_COPY_AND_ASSIGN(ExternFunctionProtoRegistry);
};

namespace detail {

FunctionProto* CreateTanhProto();
FunctionProto* CreateTanhVProto();

}  // namespace detail
}  // namespace backends
}  // namespace cinn
