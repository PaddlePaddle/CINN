#include "cinn/hlir/pe/broadcast.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForElementwiseAdd(const framework::NodeAttr &attrs,
                                                      const std::vector<ir::Tensor> &inputs,
                                                      const std::vector<Type> &out_type,
                                                      const Target &target) {
  framework::CINNCompute add_compute([&attrs](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of add compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for add compute\n";
    Expr A_expr = a[0];
    Expr B_expr = a[1];
    CHECK(A_expr.as_tensor());
    CHECK(B_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor B = B_expr.as_tensor_ref();
    Expr axis;
    bool trans_a;
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "axis") {
        axis = Expr(std::get<int>(iter.second));
      } else {
        LOG(ERROR) << "unsupported attr_store: " << iter.first << std::endl;
      }
    }

    auto out = pe::Add(A, B, UniqName("C"), axis);

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule add_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of add schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
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
    CHECK(!args.empty()) << "The input argument of elementwise_mul compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for elementwise_mul compute\n";
    Expr A_expr = a[0];
    Expr B_expr = a[1];
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

    auto stages = CreateStages({out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule mul_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of elementwise_mul schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(mul_compute, mul_schedule, "strategy.elementwise_mul.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForElementwise(const std::vector<shape_t> &inputs_shape,
                                              const framework::NodeAttr &attrs) {
  CHECK_EQ(inputs_shape.size(), 2UL);
  std::vector<shape_t> res{inputs_shape[0]};
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
    CHECK_EQ(arg_pack.size(), 2UL) << "The input tensor's size of scale schedule is " << arg_pack.size()
                                   << "and it should be equal to 2! Please check.";
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, scale_schedule, "strategy.scale.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForScale(const std::vector<shape_t> &inputs_shape, const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  return {{inputs_shape[0]}};
}

std::vector<Type> InferDtypeForScale(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<OpStrategy> StrategyForSoftmax(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const Target &target) {
  int axis = -1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "axis") {
      axis = std::get<int>(iter.second);
    }
  }
  framework::CINNCompute softmax_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of softmax compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensors of softmax compute is empty! Please check.";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor out;
    int new_axis = axis;
    if (axis == -1) {
      new_axis = A->shape.size() - 1;
    }

    Var axis_j(A->shape[new_axis], UniqName("axis_j"));
    auto temp = Compute(A->shape,
                        [=](const std::vector<Expr> &indice) {
                          std::vector<Expr> new_indice = indice;
                          new_indice[new_axis]         = axis_j;
                          return ir::ReduceSum(Exp(A(new_indice)), Expr(0.f));
                        },
                        "softmax_temp_out",
                        {axis_j});
    out       = Compute(
        A->shape, [=](const std::vector<Expr> &indice) { return Exp(A(indice)) / temp(indice); }, "softmax_out");
    auto stages = CreateStages({temp, out});
    *ret        = CINNValuePack{{CINNValue(Expr(temp.get())), CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule softmax_schedule([](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of softmax schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL) << "The input tensor's size of softmax schedule is " << arg_pack.size()
                                   << "and it should be equal to 3! Please check.";
    Expr A [[maybe_unused]] = arg_pack[0];
    *ret                    = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(softmax_compute, softmax_schedule, "strategy.softmax.x86", 1);

  return strategy;
}

std::vector<std::vector<int>> InferShapeForSoftmax(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::NodeAttr &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0], inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForSoftmax(const std::vector<Type> &inputs_type, const framework::NodeAttr &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0], inputs_type[0]};
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

  CINN_REGISTER_OP(softmax)
      .describe("This operator implements the softmax layer")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSoftmax)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForSoftmax))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForSoftmax))
      .set_support_level(4);

  return true;
}
