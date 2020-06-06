#pragma once
#include <string>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/hlir/instruction/context.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct BinaryBasicImpl {
  using opr_t = std::function<cinn::Expr(cinn::Expr, cinn::Expr)>;
  BinaryBasicImpl(Context* ctx, opr_t opr, bool inlined) : ctx_(ctx), opr_(opr), inlined_(inlined) {}

  cinn::ir::Tensor operator()(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b, const std::string& name);

  cinn::ir::Tensor RunWithArgb1Dim(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b);
  cinn::ir::Tensor RunWithArgb2Dim(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b);
  cinn::ir::Tensor RunWithArgb3Dim(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b);
  cinn::ir::Tensor RunWithArgb4Dim(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b);
  cinn::ir::Tensor RunWithArgb5Dim(const cinn::ir::Tensor& a, const cinn::ir::Tensor& b);

 private:
  Context* ctx_{};
  opr_t opr_;
  bool inlined_{false};
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
