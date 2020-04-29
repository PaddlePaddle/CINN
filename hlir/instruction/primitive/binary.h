#pragma once
#include <string>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/lang/tensor.h"
#include "hlir/instruction/context.h"

namespace hlir {
namespace instruction {
namespace primitive {

struct BinaryImpl {
  using opr_t = std::function<cinn::Expr(cinn::Expr, cinn::Expr)>;
  BinaryImpl(Context* ctx, opr_t opr, bool inlined) : ctx_(ctx), opr_(opr), inlined_(inlined) {}

  cinn::ir::Tensor operator()(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b, const std::string& name);

 private:
  Context* ctx_{};
  opr_t opr_;
  bool inlined_{false};
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
