#include "cinn/hlir/op/contrib/argmax.h"

#include <iostream>
#include <vector>

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_schedule.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using framework::shape_t;
using ir::Tensor;

Tensor Argmax(const Tensor &in_tensor, const int &axis, const bool keep_dims, const std::string &output_name) {
  auto shape = in_tensor->shape;
  auto ndim  = shape.size();
  CHECK_GT(ndim, 0) << "tensor's dim must be more than 0";

  int real_axis;
  if (axis < 0) {
    real_axis = static_cast<int>(ndim) + axis;
  } else {
    real_axis = axis;
  }
  CHECK_LT(real_axis, ndim) << "Axis must be less than tensor's dim";
  CHECK_GE(real_axis, 0) << "Axis must be more than 0";

  auto compute = [=](const std::vector<Expr> &indices) -> Expr {
    //    const Expr current[2] = {Expr(0), Expr(-3.402823e+38f)};
    std::vector<Expr> eval_indices(indices);
    if (!keep_dims) {
      eval_indices.insert(eval_indices.begin() + real_axis, Expr(1));
    }
    CHECK_EQ(eval_indices.size(), ndim);

    //    Var loop_var("k0");
    //    eval_indices[real_axis] = i;
    //    auto value = in_tensor(eval_indices);
    //    auto update = ir::LT::Make(value, current[1]);
    //    auto c1 = ir::Select::Make(update, Expr(i), current[0]);
    //    auto c2 = ir::Select::Make(update, value, current[1]);
    //    current[0] = c1;
    //    current[1] = c2;
    //    auto for_loop = ir::For::Make(i, Expr(0), current[0]);

    //    Placeholder<float> max_value("max_value", {shape[real_axis]});
    //    Placeholder<float> max_index("max_index", {shape[real_axis]});
    //    Var loop_var("k0", Int(32));
    //    eval_indices[real_axis] = Expr(loop_var);
    //    auto value = in_tensor(eval_indices);
    //    auto update = ir::LT::Make(value, ir::Load::Make(ir::Tensor(max_value), {Expr(loop_var)}));
    //    auto c_v = ir::Select::Make(update, value, ir::Load::Make(ir::Tensor(max_value), {Expr(loop_var)}));
    //    auto c_i = ir::Select::Make(update, Expr(loop_var), ir::Load::Make(ir::Tensor(max_index), {Expr(loop_var)}));
    //
    //    Expr body1 = ir::Store::Make(ir::Tensor(max_value), c_v, {Expr(loop_var)+1});
    //    Expr body2 = ir::Store::Make(ir::Tensor(max_index), c_i, {Expr(loop_var)+1});
    //
    //    Expr body = ir::Block::Make({body1, body2});
    //
    //    auto output = ir::For::Make(loop_var, common::make_const(0), shape[real_axis]-1,
    //                          ir::ForType::Serial, ir::DeviceAPI::Host, body);

    //    for (int i = 0; i<shape[real_axis]; i++){
    //    }

    return in_tensor(eval_indices);
    //    return ir::Load::Make(output, {shape[real_axis]-1});
  };

  std::vector<Expr> output_shape(shape);
  if (keep_dims) {
    output_shape.at(real_axis) = Expr(1);
  } else {
    output_shape.erase(output_shape.begin() + real_axis);
  }
  if (output_shape.empty()) {
    output_shape.push_back(Expr(1));
  }

  Tensor res = Compute(output_shape, compute, output_name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForArgmax(const framework::NodeAttr &attrs,
                                                         const std::vector<Tensor> &inputs,
                                                         const std::vector<Type> &out_type,
                                                         const std::vector<std::vector<int>> &output_shapes,
                                                         const Target &target) {
  int axis;
  bool keep_dims = false;

  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  } else {
    LOG(FATAL) << "reduce dimension is not set!";
  }
  if (attrs.attr_store.count("keep_dim")) {
    keep_dims = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }

  framework::CINNCompute argmax_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of argmax compute is empty! Please check.";
    common::CINNValuePack arg_packs = args[0];
    std::string tensor_name         = UniqName("T_Argmax_out");
    CHECK_EQ(arg_packs.size(), 1U) << "There should be 1 input args for argmax compute";
    Expr in_expr = arg_packs[0];
    CHECK(in_expr.as_tensor());
    Tensor in_tensor = in_expr.as_tensor_ref();
    auto out         = Argmax(in_tensor, axis, keep_dims, tensor_name);
    auto stages      = CreateStages({out});

    std::vector<CINNValue> cinn_values{CINNValue(out), CINNValue(stages)};
    *ret = common::CINNValuePack{cinn_values};
  });

  framework::CINNSchedule argmax_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of argmax schedule is empty! Please check.";
    common::CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 2UL);
    Expr out = arg_pack[0];
    CHECK(out.as_tensor());

    // When develop FLAGS_cinn_ir_schedule=true case, we should run unit test with
    // FLAGS_cinn_ir_schedule=1
    if (FLAGS_cinn_ir_schedule) {
      *ret = common::CINNValuePack{{common::CINNValue(out)}};
    } else {
      poly::StageMap stages = arg_pack[arg_pack.size() - 1];
      *ret                  = common::CINNValuePack{{common::CINNValue(out), common::CINNValue(stages)}};
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(argmax_compute, argmax_schedule, "strategy.argmax.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForArgmax(const std::vector<shape_t> &inputs_shape,
                                         const framework::AttrMapType &attrs) {
  CHECK(inputs_shape.size() == 1UL);
  auto ndim = inputs_shape[0].size();
  CHECK_GT(ndim, 0) << "tensor's dim must be more than 0";
  int axis;
  bool keep_dim;

  CHECK(attrs.find("axis") != attrs.end());
  axis = absl::get<int>(attrs.at("axis"));
  if (axis < 0) {
    axis = static_cast<int>(ndim) + axis;
  }
  CHECK_LT(axis, ndim) << "Axis must be less than tensor's dim";
  CHECK_GE(axis, 0) << "Axis must be more than 0";

  CHECK(attrs.find("keep_dim") != attrs.end());
  keep_dim = absl::get<bool>(attrs.at("keep_dim"));

  std::vector<int> out_shapes(inputs_shape[0]);
  for (size_t i = 0; i < ndim; ++i) {
    if (axis == i) {
      if (keep_dim) {
        out_shapes.push_back(1);
      }
    } else {
      out_shapes.push_back(inputs_shape[0][i]);
    }
  }

  if (keep_dim) {
    CHECK_EQ(ndim, out_shapes.size());
  } else {
    CHECK_EQ(ndim - 1, out_shapes.size());
  }
  if (out_shapes.empty()) {
    out_shapes.push_back(1);
  }

  return {out_shapes};
}

std::vector<Type> InferDtypeForArgmax(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForArgmax(const std::vector<framework::shape_t> &input_shapes,
                                                           const std::vector<std::string> &input_layouts,
                                                           const framework::NodeAttr &attrs,
                                                           const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> new_input_layouts = input_layouts;
  if (input_shapes[0].size() > 4) {
    // alter input layout back
    new_input_layouts[0] = "NCHW";
    VLOG(3) << "alter input layout from " << input_layouts[0] << " to " << new_input_layouts[0];
  }

  return {{""}, new_input_layouts};
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(argmax_ops) {
  CINN_REGISTER_OP(argmax)
      .describe("This operator implements the op argmax.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForArgmax)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForArgmax))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForArgmax))
      .set_support_level(4);

  return true;
}
