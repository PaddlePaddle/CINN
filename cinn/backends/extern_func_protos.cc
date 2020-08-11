#include "cinn/backends/extern_func_protos.h"

namespace cinn {
namespace backends {

ExternFunctionProtoRegistry::ExternFunctionProtoRegistry() {
  static const std::vector<std::string> extern_funcs_fp32 = {
      "exp",      "erf",   "sigmoid",    "sqrt",        "log",        "log2",        "log10",       "floor",
      "ceil",     "round", "trunc",      "cos",         "cosh",       "tan",         "sin",         "sinh",
      "acos",     "acosh", "asin",       "asinh",       "atan",       "atanh",       "isnan",       "tanh",
      "isfinite", "isinf", "left_shift", "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"};
  static const std::vector<std::string> extern_funcs_int64 = {
      "left_shift", "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"};
  for (int i = 0; i < extern_funcs_fp32.size(); ++i) {
    auto* proto = new FunctionProto(extern_funcs_fp32[i], {Float(32)}, Float(32));
    Register(proto->name, proto);
  }
  for (int i = 0; i < extern_funcs_int64.size(); ++i) {
    auto* proto = new FunctionProto(extern_funcs_int64[i], {Int(64)}, Int(64));
    Register(proto->name, proto);
  }
  auto* n = detail::CreateTanhVProto();
  Register(n->name, n);
}

ExternFunctionProtoRegistry& ExternFunctionProtoRegistry::Global() {
  static ExternFunctionProtoRegistry x;
  return x;
}

namespace detail {

FunctionProto* CreateTanhVProto() {
  return new FunctionProto(
      extern_func__tanh_v, {type_of<float*>()}, {type_of<float*>()}, Void(), FunctionProto::ShapeFollowNthArgument(0));
}

}  // namespace detail
}  // namespace backends
}  // namespace cinn
