#include "cinn/hlir/pe/broadcast.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForElementwiseAdd(const framework::NodeAttr &attrs,
                                                      const std::vector<ir::Tensor> &inputs,
                                                      const std::vector<Type> &out_type,
                                                      const Target &target) {
  framework::CINNCompute add_compute([&attrs](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    Expr A_expr     = a[0];
    Expr B_expr     = a[1];
    CHECK(A_expr.as_tensor());
    CHECK(B_expr.as_tensor());
    ir::Tensor A    = A_expr.as_tensor_ref();
    ir::Tensor B    = B_expr.as_tensor_ref();
    auto attr_store = attrs.attr_store;
    auto iter       = attr_store.find("axis");
    Expr axis;
    if (iter != attr_store.end()) {
      axis = Expr(std::get<int>(iter->second));
    }

    auto out = pe::Add(A, B, UniqName("C"), axis);

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule add_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(add_compute, add_schedule, "strategy.elementwise_add.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForElementwiseMul(const framework::NodeAttr &attrs,
                                                      const std::vector<ir::Tensor> &inputs,
                                                      const std::vector<Type> &out_type,
                                                      const Target &target) {
  framework::CINNCompute mul_compute([&attrs](lang::Args args, lang::RetValue *ret) {
    CINNValuePack a = args[0];
    Expr A_expr     = a[0];
    Expr B_expr     = a[1];
    CHECK(A_expr.as_tensor());
    CHECK(B_expr.as_tensor());
    ir::Tensor A    = A_expr.as_tensor_ref();
    ir::Tensor B    = B_expr.as_tensor_ref();
    auto attr_store = attrs.attr_store;
    auto iter       = attr_store.find("axis");
    Expr axis;
    if (iter != attr_store.end()) {
      axis = Expr(std::get<int>(iter->second));
    }

    auto out = pe::Multiply(A, B, UniqName("C"), axis);
    LOG(INFO) << "Multiply out " << out;

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule mul_schedule([](lang::Args args, lang::RetValue *ret) {
    CINNValuePack arg_pack  = args[0];
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.elementwise_mul.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForElementwise(const std::vector<std::vector<int>> &inputs_shape,
                                                       const framework::NodeAttr &attrs) {
  CHECK_EQ(inputs_shape.size(), 2UL);
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForElementwise(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForScale(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const Target &target) {
  float scale           = 1.f;
  float bias            = 0.f;
  bool bias_after_scale = true;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "scale") {
      scale = std::get<float>(iter.second);
    } else if (iter.first == "bias") {
      bias = std::get<float>(iter.second);
    } else if (iter.first == "bias_after_scale") {
      bias_after_scale = std::get<bool>(iter.second);
    }
  }
  framework::CINNCompute scale_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of scale compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensors of scale compute is empty! Please check.";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor out;
    if (bias_after_scale) {
      out = Compute(
          A->shape, [=](const std::vector<Expr> &indice) { return scale * A(indice) + bias; }, UniqName("Scale_out"));
    } else {
      out = Compute(
          A->shape, [=](const std::vector<Expr> &indice) { return scale * (A(indice) + bias); }, UniqName("Scale_out"));
    }
    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule scale_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of scale schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK(!arg_pack.empty()) << "The input tensor of scale schedule is empty! Please check.";
    Expr A [[maybe_unused]] = arg_pack[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, scale_schedule, "strategy.scale.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForScale(const std::vector<std::vector<int>> &inputs_shape,
                                                 const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForScale(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(broadcast_ops) {
  CINN_REGISTER_OP(elementwise_add)
      .describe("Add two tensors")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForElementwiseAdd)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForElementwise))
      .set_support_level(4);

  CINN_REGISTER_OP(elementwise_mul)
      .describe("multiply two tensors")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForElementwiseMul)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForElementwise))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForElementwise))
      .set_support_level(4);

  CINN_REGISTER_OP(scale)
      .describe("Putting scale and bias to the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForScale)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForScale))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForScale))
      .set_support_level(4);

  return true;
}
