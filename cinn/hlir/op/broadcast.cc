#include "cinn/hlir/pe/broadcast.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/layout.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using namespace pe;

#define StrategyForBinary(op_name__, pe__)                                                          \
  std::shared_ptr<OpStrategy> StrategyFor##pe__(const framework::NodeAttr &attrs,                   \
                                                const std::vector<ir::Tensor> &inputs,              \
                                                const std::vector<Type> &out_type,                  \
                                                const std::vector<std::vector<int>> &output_shapes, \
                                                const Target &target) {                             \
    return StrategyForBroadcast(attrs, inputs, out_type, output_shapes, target, #op_name__, pe__);  \
  }

std::shared_ptr<OpStrategy> StrategyForBroadcast(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    ir::Tensor (*pe_func)(const ir::Tensor &A, const ir::Tensor &B, const std::string &output_name, const Expr &axis)) {
  framework::CINNCompute binary_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 2U) << "at least 2 input tensors for " << op_name << " compute";
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
        axis = Expr(absl::get<int>(iter.second));
        break;
      }
    }
    auto out    = pe_func(A, B, UniqName(op_name + "_Out"), axis);
    auto stages = CreateStages({A, B, out});
    *ret        = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
  });

  framework::CINNSchedule binary_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    if (target.arch == Target::Arch::NVGPU) {
      Expr Out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(Out.as_tensor());
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.back(), target);
    } else if (target.arch == Target::Arch::X86) {
      Expr Out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(Out.as_tensor());
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.back(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(binary_compute, binary_schedule, "strategy." + op_name + ".x86", 1);
  return strategy;
}

std::vector<shape_t> InferShapeForBroadcast(const std::vector<shape_t> &inputs_shape,
                                            framework::NodeAttr &attrs,
                                            const Target &target) {
  CHECK_EQ(inputs_shape.size(), 2UL);
  std::vector<int> out_shape;
  int axis = -1;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
      break;
    }
  }
  pe::GetBroadcastOutShape(inputs_shape[0], inputs_shape[1], &out_shape, axis);
  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  return {out_shape};
}

std::vector<Type> InferDtypeForBroadcast(const std::vector<Type> &inputs_type,
                                         const framework::NodeAttr &attrs,
                                         const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForBroadcast(const std::vector<std::vector<int>> &input_shapes,
                                                              const std::vector<std::string> &input_layouts,
                                                              const framework::NodeAttr &attrs,
                                                              const Target &target) {
  int input_size = input_layouts.size();
  CHECK(input_size == 2U || input_size == 3U) << "The input's layouts size is not 2 or 3! Please check again.";
  int axis = -1;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  std::vector<std::string> out_layouts = input_layouts;
  if (input_layouts[0].empty() && input_layouts[1].empty()) {
    return {{input_layouts[0]}, input_layouts};
  } else if (input_layouts[0].empty() || input_layouts[1].empty()) {
    int undef_idx = input_layouts[0] == "" ? 0 : 1;
    int def_idx   = 1 - undef_idx;
    CHECK_GE(input_shapes[def_idx].size(), input_shapes[undef_idx].size());
    auto ret = out_layouts[def_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  } else {
    // e.g. NCHWxc + NCHW
    ir::Layout layout0(input_layouts[0]);
    ir::Layout layout1(input_layouts[1]);
    int large_idx = layout0.ndims() >= layout1.ndims() ? 0 : 1;
    auto ret      = input_layouts[large_idx];
    if (input_size == 2) {
      return {{ret}, {ret, ret}};
    } else {
      return {{ret}, {ret, ret, ret}};
    }
  }
}

std::shared_ptr<OpStrategy> StrategyForScale(const framework::NodeAttr &attrs,
                                             const std::vector<ir::Tensor> &inputs,
                                             const std::vector<Type> &out_type,
                                             const std::vector<std::vector<int>> &output_shapes,
                                             const Target &target) {
  float scale           = 1.f;
  float bias            = 0.f;
  bool bias_after_scale = true;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "scale") {
      scale = absl::get<float>(iter.second);
    } else if (iter.first == "bias") {
      bias = absl::get<float>(iter.second);
    } else if (iter.first == "bias_after_scale") {
      bias_after_scale = absl::get<bool>(iter.second);
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

  framework::CINNSchedule scale_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of scale schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL) << "The input tensor's size of scale schedule is " << arg_pack.size()
                                   << "and it should be equal to 2! Please check.";
    if (target.arch == Target::Arch::NVGPU) {
      Expr Out              = arg_pack[0];
      poly::StageMap stages = arg_pack[1];
      CHECK(Out.as_tensor());
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.back(), target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(scale_compute, scale_schedule, "strategy.scale.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForScale(const std::vector<shape_t> &inputs_shape,
                                        framework::NodeAttr &attrs,
                                        const Target &target) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  return {{inputs_shape[0]}};
}

std::vector<Type> InferDtypeForScale(const std::vector<Type> &inputs_type,
                                     framework::NodeAttr &attrs,
                                     const Target &target) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForScale(const std::vector<framework::shape_t> &input_shapes,
                                                          const std::vector<std::string> &input_layouts,
                                                          const framework::NodeAttr &attrs,
                                                          const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}

StrategyForBinary(elementwise_add, Add);
StrategyForBinary(elementwise_mul, Multiply);

StrategyForBinary(bitwise_or, BitwiseOr);
StrategyForBinary(bitwise_xor, BitwiseXor);
StrategyForBinary(bitwise_and, BitwiseAnd);
StrategyForBinary(left_shift, LeftShift);
StrategyForBinary(right_shift, RightShift);

#undef StrategyForBinary

}  // namespace op
}  // namespace hlir
}  // namespace cinn


template <typename R, typename ...Args>
inline auto make_function(R(*f)(Args...)) {
  return std::function<R(Args...)>(f);
}

CINN_REGISTER_HELPER(broadcast_ops) {
#define CINN_REGISTER_BINARY(op__, op_stragegy__)                                                                    \
  CINN_REGISTER_OP(op__)                                                                                             \
      .describe(#op__ " function")                                                                                   \
      .set_num_inputs(1)                                                                                             \
      .set_num_outputs(1)                                                                                            \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__) \
      .set_attr("infershape", make_function(cinn::hlir::op::InferShapeForBroadcast))                                 \
      .set_attr("inferdtype", make_function(cinn::hlir::op::InferDtypeForBroadcast))                                 \
      .set_attr("inferlayout", make_function(cinn::hlir::op::InferLayoutForBroadcast))                               \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast) \
      .set_support_level(4);

  CINN_REGISTER_BINARY(elementwise_add, Add);
  CINN_REGISTER_BINARY(elementwise_mul, Multiply);

  CINN_REGISTER_BINARY(bitwise_or, BitwiseOr);
  CINN_REGISTER_BINARY(bitwise_xor, BitwiseXor);
  CINN_REGISTER_BINARY(bitwise_and, BitwiseAnd);
  CINN_REGISTER_BINARY(left_shift, LeftShift);
  CINN_REGISTER_BINARY(right_shift, RightShift);
#undef CINN_REGISTER_BINARY

  CINN_REGISTER_OP(scale)
      .describe("Putting scale and bias to the input Tensor")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForScale)
      .set_attr("infershape", make_function(cinn::hlir::op::InferShapeForScale))
      .set_attr("inferdtype", make_function(cinn::hlir::op::InferDtypeForBroadcast))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", make_function(cinn::hlir::op::InferLayoutForScale))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)
      .set_support_level(4);

  return true;
}
