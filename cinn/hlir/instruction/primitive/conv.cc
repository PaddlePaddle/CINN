#include "cinn/hlir/instruction/primitive/conv.h"

#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/lower.h"
#include "cinn/hlir/instruction/lower_impl.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

// This implementation is borrowed from TVM
ir::Tensor Pad(ir::Tensor t,
               const std::vector<Expr>& pad_before,
               std::vector<Expr> pad_after,
               Expr pad_value,
               const std::string& name,
               const std::string& pad_mode) {
  if (pad_after.size() < pad_before.size()) {
    for (size_t i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }

  CHECK_GE(pad_before.size(), 1UL);
  CHECK_EQ(pad_before.size(), pad_after.size());

  std::vector<Expr> output_shape;

  for (size_t i = 0; i < t->shape.size(); ++i) {
    if (i >= pad_before.size()) {
      output_shape.push_back(t->shape[i]);
    } else {
      output_shape.push_back(t->shape[i] + pad_before[i] + pad_after[i]);
    }
  }

  if (!pad_value.defined()) {
    pad_value = common::make_const(t->type(), 0);
  }

  auto l = [=](const std::vector<Expr>& ovars) -> Expr {
    std::vector<Expr> indices, sel, pad_idx;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        indices.push_back(ovars[i]);
        continue;
      }

      if (!common::is_zero(pad_before[i])) {
        sel.push_back(ovars[i] >= pad_before[i]);
        indices.push_back(ovars[i] - pad_before[i]);
      } else {
        indices.push_back(ovars[i]);
      }
      if (!common::is_zero(pad_after[i])) {
        sel.push_back(ovars[i] < pad_before[i] + t->shape[i]);
      }
      if (pad_mode == "edge") {
        pad_idx.push_back(ir::Select::Make(
            Expr(ovars[i]) < pad_before[i],
            common::make_const(Int(32), 0),
            ir::Select::Make(ovars[i] >= pad_before[i] + t->shape[i], t->shape[i] - 1, ovars[i] - pad_before[i])));
      } else if (pad_mode == "reflect") {
        pad_idx.push_back(ir::Select::Make(Expr(ovars[i]) < pad_before[i],
                                           pad_before[i] - ovars[i],
                                           ir::Select::Make(ovars[i] >= pad_before[i] + t->shape[i],
                                                            t->shape[i] * 2 - ovars[i] + pad_before[i] - 2,
                                                            ovars[i] - pad_before[i])));
      }
    }

    if (!sel.empty()) {
      auto cond = sel[0];
      for (int i = 1; i < sel.size(); i++) {
        cond = ir::And::Make(cond, sel[i]);
      }

      if (pad_mode == "constant") {
        return ir::Select::Make(cond, t(indices), pad_value);
      } else if (pad_mode == "edge" || pad_mode == "reflect") {
        return ir::Select::Make(cond, t(indices), t(pad_idx));
      }
    }

    return t(indices);
  };

  return Compute(output_shape, l, name);
}

// This implementation is borrowed from TVM
ir::Tensor Conv2dNCHW(
    ir::Tensor I, ir::Tensor W, int pad_h, int pad_w, int stride_h, int stride_w, const std::string& name) {
  CHECK_EQ(I->shape.size(), 4UL);
  CHECK_EQ(W->shape.size(), 4UL);
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  std::vector<Expr> output_shape({I->shape[0],
                                  W->shape[0],
                                  (I->shape[2] - W->shape[2] + 2 * pad_h) / stride_h + 1,
                                  (I->shape[3] - W->shape[3] + 2 * pad_w) / stride_w + 1});

  Var i(common::make_zero(), I->shape[1], "i0");
  Var kh(common::make_zero(), I->shape[2], "kh");
  Var kw(common::make_zero(), I->shape[3], "kw");
  auto T = (pad_h == 0 && pad_w == 0) ? I : Pad(I, {Expr(0), Expr(0), Expr(pad_h), Expr(pad_w)});

  auto l = [=](Var b, Var o, Var h, Var w) -> Expr {
    return Sum(T(b, i, stride_h * h + kh, stride_w * w + kw) * W(o, i, kh, kw));
  };

  return Compute(output_shape, l, name, {i, kh, kw});
}

struct Conv2dNCHWLowerImpl : public LowerImplBase {
  explicit Conv2dNCHWLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(Instruction* instr, Context* context, ModuleLower* module_lower) override {
    CHECK_EQ(instr->operand_count(), 2UL) << "Conv2dNCHW should take two arguments";
    Expr I = module_lower->scope().Lookup(instr->operand(0));
    Expr W = module_lower->scope().Lookup(instr->operand(1));
    CHECK(I.defined());
    CHECK(W.defined());
    auto* conv_instr = instr->As<Conv>();
    auto out         = Conv2dNCHW(I.as_tensor_ref(),
                          W.as_tensor_ref(),
                          conv_instr->pad_h(),
                          conv_instr->pad_w(),
                          conv_instr->stride_h(),
                          conv_instr->stride_w());
    module_lower->scope().Insert(instr, out);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
