#include "cinn/backends/extern_func_protos.h"

namespace cinn {
namespace backends {

ExternFunctionProtoRegistry::ExternFunctionProtoRegistry() {
  {
    auto* n = detail::CreateTanhProto();
    Register(n->name, n);
  }
  {
    auto* n = detail::CreateTanhVProto();
    Register(n->name, n);
  }

  Register("cos", new FunctionProto("cos", {Float(32)}, Float(32)));
  Register("sign", new FunctionProto("sign", {Float(32)}, Float(32)));
  Register("sin", new FunctionProto("sin", {Float(32)}, Float(32)));
  Register("tanh", new FunctionProto("tanh", {Float(32)}, Float(32)));
  Register("log", new FunctionProto("log", {Float(32)}, Float(32)));
}

ExternFunctionProtoRegistry& ExternFunctionProtoRegistry::Global() {
  static ExternFunctionProtoRegistry x;
  return x;
}

namespace detail {

FunctionProto* CreateTanhProto() { return new FunctionProto(extern_func__tanh, {Float(32)}, {}, Float(32)); }
FunctionProto* CreateTanhVProto() {
  return new FunctionProto(
      extern_func__tanh_v, {type_of<float*>()}, {type_of<float*>()}, Void(), FunctionProto::ShapeFollowNthArgument(0));
}

}  // namespace detail
}  // namespace backends
}  // namespace cinn
