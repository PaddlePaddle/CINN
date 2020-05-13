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
}

namespace detail {

FunctionProto* CreateTanhProto() { return new FunctionProto(extern_func__tanh, {Float(32)}, {}, Float(32)); }
FunctionProto* CreateTanhVProto() {
  return new FunctionProto(
      extern_func__tanh_v, {Float(32)}, {type_of<float*>()}, Void(), FunctionProto::ShapeFollowNthArgument(0));
}

}  // namespace detail
}  // namespace backends
}  // namespace cinn
