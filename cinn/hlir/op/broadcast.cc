#include "cinn/hlir/pe/broadcast.h"

#include <iostream>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
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
        axis = Expr(std::get<int>(iter.second));
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
    Expr Out              = arg_pack[0];
    poly::StageMap stages = arg_pack[1];
    CHECK(Out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], output_shapes.front(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], output_shapes.front(), target);
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
      axis = std::get<int>(iter.second);
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
    axis = std::get<int>(attrs.attr_store.at("axis"));
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

std::shared_ptr<OpStrategy> StrategyForBroadcastTo(const framework::NodeAttr &attrs,
                                                   const std::vector<ir::Tensor> &inputs,
                                                   const std::vector<Type> &out_type,
                                                   const std::vector<std::vector<int>> &output_shapes,
                                                   const Target &target) {
  std::vector<int> out_shape;
  std::vector<int> broadcast_axes;
  if (attrs.attr_store.count("out_shape")) {
    out_shape = std::get<std::vector<int>>(attrs.attr_store.at("out_shape"));
  }
  if (attrs.attr_store.count("broadcast_axes")) {
    broadcast_axes = std::get<std::vector<int>>(attrs.attr_store.at("broadcast_axes"));
  }

  framework::CINNCompute broadcast_to_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of broadcast_to compute is empty! Please check.";
    CINNValuePack a = args[0];
    CHECK(!a.empty()) << "The input tensors of broadcast_to compute is empty! Please check.";
    Expr A_expr = a[0];
    CHECK(A_expr.as_tensor());
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out     = BroadcastTo(A, out_shape, broadcast_axes, UniqName("broadcast_to_Out"));
    auto stages  = CreateStages({A, out});
    *ret         = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
  });

  framework::CINNSchedule broadcast_to_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of broadcast_to schedule is empty! Please check.";
    CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr Out              = arg_pack[0];
    poly::StageMap stages = arg_pack.back();
    CHECK(Out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[Out.as_tensor_ref()], out_shape, target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[Out.as_tensor_ref()], out_shape, target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(broadcast_to_compute, broadcast_to_schedule, "strategy.broadcast_to.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForBroadcastTo(const std::vector<shape_t> &inputs_shape,
                                              framework::NodeAttr &attrs,
                                              const Target &target) {
  CHECK_EQ(inputs_shape.size(), 1UL) << "input_shape size should be one. Please Check.";
  std::vector<int> broadcast_axes;
  std::vector<int> out_shape;
  CHECK(attrs.attr_store.count("broadcast_axes"));
  CHECK(attrs.attr_store.count("out_shape"));
  out_shape      = std::get<std::vector<int>>(attrs.attr_store.at("out_shape"));
  broadcast_axes = std::get<std::vector<int>>(attrs.attr_store.at("broadcast_axes"));

  CHECK_EQ(inputs_shape[0].size(), broadcast_axes.size())
      << "broadcast_axes's size should be same with the input shape's size";
  CHECK_GE(out_shape.size(), broadcast_axes.size()) << "broadcast_axes's size should be no more than out_shape's size";

  VLOG(3) << "broadcast out shape: " << utils::Join(out_shape, ", ");
  return {out_shape};
}

std::vector<std::vector<std::string>> InferLayoutForBroadcastTo(const std::vector<std::vector<int>> &input_shapes,
                                                                const std::vector<std::string> &input_layouts,
                                                                const framework::NodeAttr &attrs,
                                                                const Target &target) {
  CHECK(input_layouts.size() == 1U) << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> out_layouts = {""};
  if (attrs.attr_store.count("out_layouts")) {
    out_layouts = std::get<std::vector<std::string>>(attrs.attr_store.at("out_layouts"));
  }
  return {out_layouts, input_layouts};
}

StrategyForBinary(elementwise_add, Add);
StrategyForBinary(elementwise_mul, Multiply);

StrategyForBinary(substract, Substract);
StrategyForBinary(divide, Divide);
StrategyForBinary(floor_divide, FloorDivide);
StrategyForBinary(mod, Mod);
StrategyForBinary(floor_mod, FloorMod);
StrategyForBinary(max, Maximum);
StrategyForBinary(min, Minimum);
StrategyForBinary(power, Power);
StrategyForBinary(logical_and, LogicaAnd);
StrategyForBinary(logical_or, LogicalOr);
StrategyForBinary(logical_xor, LogicalXOr);
StrategyForBinary(greater, Greater);
StrategyForBinary(less, Less);
StrategyForBinary(equal, Equal);
StrategyForBinary(not_equal, NotEqual);
StrategyForBinary(greater_equal, GreaterEqual);
StrategyForBinary(less_equal, LessEqual);

StrategyForBinary(bitwise_or, BitwiseOr);
StrategyForBinary(bitwise_xor, BitwiseXor);
StrategyForBinary(bitwise_and, BitwiseAnd);
StrategyForBinary(left_shift, LeftShift);
StrategyForBinary(right_shift, RightShift);

#undef StrategyForBinary

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(broadcast_ops) {
#define CINN_REGISTER_BINARY(op__, op_stragegy__)                                                                    \
  CINN_REGISTER_OP(op__)                                                                                             \
      .describe(#op__ " function")                                                                                   \
      .set_num_inputs(1)                                                                                             \
      .set_num_outputs(1)                                                                                            \
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__) \
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForBroadcast))                                 \
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForBroadcast))                                 \
      .set_attr("inferlayout", std::function(cinn::hlir::op::InferLayoutForBroadcast))                               \
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kBroadcast) \
      .set_support_level(4);

  CINN_REGISTER_BINARY(elementwise_add, Add);
  CINN_REGISTER_BINARY(elementwise_mul, Multiply);

  CINN_REGISTER_BINARY(substract, Substract);
  CINN_REGISTER_BINARY(divide, Divide);
  CINN_REGISTER_BINARY(floor_divide, FloorDivide);
  CINN_REGISTER_BINARY(mod, Mod);
  CINN_REGISTER_BINARY(floor_mod, FloorMod);
  CINN_REGISTER_BINARY(max, Maximum);
  CINN_REGISTER_BINARY(min, Minimum);
  CINN_REGISTER_BINARY(power, Power);
  CINN_REGISTER_BINARY(logical_and, LogicaAnd);
  CINN_REGISTER_BINARY(logical_or, LogicalOr);
  CINN_REGISTER_BINARY(logical_not, LogicalXOr);
  CINN_REGISTER_BINARY(greater, Greater);
  CINN_REGISTER_BINARY(less, Less);
  CINN_REGISTER_BINARY(equal, Equal);
  CINN_REGISTER_BINARY(not_equal, NotEqual);
  CINN_REGISTER_BINARY(greater_equal, GreaterEqual);
  CINN_REGISTER_BINARY(less_equal, LessEqual);

  CINN_REGISTER_BINARY(bitwise_or, BitwiseOr);
  CINN_REGISTER_BINARY(bitwise_xor, BitwiseXor);
  CINN_REGISTER_BINARY(bitwise_and, BitwiseAnd);
  CINN_REGISTER_BINARY(left_shift, LeftShift);
  CINN_REGISTER_BINARY(right_shift, RightShift);
#undef CINN_REGISTER_BINARY

  CINN_REGISTER_OP(broadcast_to)
      .describe("broadcast one tensor to the target shape")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForBroadcastTo)
      .set_attr("infershape", std::function(cinn::hlir::op::InferShapeForBroadcastTo))
      .set_attr("inferdtype", std::function(cinn::hlir::op::InferDtypeForBroadcast))
#ifndef CINN_WITH_CUDA
      .set_attr("inferlayout", std::function(cinn::hlir::op::InferLayoutForBroadcastTo))
#endif
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kElemWise)
      .set_support_level(4);

  return true;
}
