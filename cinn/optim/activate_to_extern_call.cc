#include "cinn/optim/activate_to_extern_call.h"

#include "cinn/cinn.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace optim {

void ActivateToExternCall(Expr *e) {
  struct Mutator : ir::IRMutator<Expr *> {
    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Activate *op, Expr *expr) override {
      auto *node = expr->As<ir::Activate>();
      operator()(&node->operand(0));

      switch (node->kind) {
#define __(code__, func__)                                 \
  case ir::Activate::Kind::code__:                         \
    *expr = lang::CallExtern(#func__, {node->operand(0)}); \
    break;
        __(kTanh, cinn_cpu_tanh_fp32)
        __(kCeil, cinn_cpu_ceil_fp32)
        __(kFloor, cinn_cpu_floor_fp32)
        __(kExp, cinn_cpu_exp_fp32)
#undef __
        default:
          CINN_NOT_IMPLEMENTED
      }
    }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      CHECK(node);
      static const std::vector<std::string> extern_funcs_fp32 = {
          "exp",         "erf",         "sigmoid",     "sqrt",        "log",        "log2",        "log10",
          "floor",       "ceil",        "round",       "trunc",       "cos",        "cosh",        "tan",
          "sin",         "sinh",        "acos",        "acosh",       "asin",       "asinh",       "atan",
          "atanh",       "isnan",       "tanh",        "isfinite",    "isinf",      "left_shift",  "right_shift",
          "bitwise_or",  "bitwise_and", "bitwise_xor", "bitwise_not", "left_shift", "right_shift", "bitwise_or",
          "bitwise_and", "bitwise_xor", "bitwise_not"};
      static const std::vector<std::string> extern_funcs_int64 = {
          "left_shift", "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"};

      auto it = std::find(extern_funcs_fp32.begin(), extern_funcs_fp32.end(), node->name);
      if (it != extern_funcs_fp32.end()) {
        *expr = lang::CallExtern("cinn_cpu_" + *it + "_fp32", node->read_args);
      } else {
        it = std::find(extern_funcs_int64.begin(), extern_funcs_int64.end(), node->name);
        if (it != extern_funcs_int64.end()) {
          *expr = lang::CallExtern("cinn_cpu_" + *it + "_int64", node->read_args);
        }
      }
    }
  };

  Mutator()(e);
}

}  // namespace optim
}  // namespace cinn
