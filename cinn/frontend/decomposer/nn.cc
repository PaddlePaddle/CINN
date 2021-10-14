
#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {
namespace decomposer {

void batch_norm_train(const Instruction& instr, const DecomposerContext& context) {
  auto& x            = instr->inputs(0);
  auto& scale        = instr->inputs(1);
  auto& bias         = instr->inputs(2);
  auto& running_mean = instr->inputs(3);
  auto& running_var  = instr->inputs(4);

  auto& epsilon        = instr.GetAttrs<float>("epsilon");
  auto& layout         = instr.GetAttrs<std::string>("layout");
  auto& average_factor = instr.GetAttrs<float>("factor");

  CinnBuilder* builder_ = nullptr;
  std::vector<int> dim  = {};
  float num_element     = 0;
  int channel           = 0;
  int c_dim             = 0;
  if (layout == "NCHW") {
    c_dim       = 1;
    dim         = {0, 2, 3};
    channel     = x->shape[1];
    num_element = x->shape[0] * x->shape[2] * x->shape[3];
  } else if (layout == "NHWC") {
    c_dim       = 3;
    dim         = {0, 1, 2};
    channel     = x->shape[3];
    num_element = x->shape[0] * x->shape[1] * x->shape[2];
  } else {
    LOG(FATAL) << layout << " setting is not support!";
  }

  auto v_num     = builder->ConstScalar<float>(num_element);
  auto v_epsilon = builder->ConstScalar<float>(epsilon);

  // compute sum
  auto sum = builder->Reduce(x, ReduceKind::kSum, dim);
  // compute mean
  auto save_mean = builder->Div(sum, v_num);
  auto mean      = builder->BroadcastTo(save_mean, x->shape, {c_dim});
  // diff
  auto diff = builder->Sub(x, mean);
  // diff2
  auto diff2 = builder->Mul(diff, diff);

  // sum variance
  auto sum_var2 = builder->Reduce(diff2, ReduceKind::kSum, dim);
  // variance
  auto var2 = builder->Div(sum_var2, v_num);
  // standard variance
  auto save_var = builder->Add(builder->Sqrt(var2), v_epsilon);
  auto var      = builder->BroadcastTo(save_var, x->shape, {c_dim});

  auto scale_  = builder->BroadcastTo(scale, x->shape, {c_dim});
    auto bias_ = builder->BroadcastTo(bias, x->shape, {c_dim]});
    // (x - mean)/var * scale + bias
    auto y = builder->Add(bias_, builder->Mul(scale_, builder->Div(diff, var)));

    auto factor   = builder->ConstScalar<float>(average_factor);
    auto factor_  = builder->ConstScalar<float>(1.0f - average_factor);
    auto new_mean = builder->Add(builder->Mul(running_mean, factor), builder->Mul(save_mean, factor_));
    auto new_var  = builder->Add(builder->Mul(running_var, factor), builder->Mul(save_var, factor_));

    // map output id
    y.set_id(instr->outputs(0)->id);
    save_mean.set_id(instr->outputs(1)->id);
    save_var.set_id(instr->outputs(2)->id);
    new_mean.set_id(instr->output(3)->id);
    new_var.set_id(instr->output(4)->id);
}

void batch_norm_grad(const Instruction* instr, const DecomposerContext& context) {
  auto& x         = instr->inputs(0);
  auto& dy        = instr->inputs(1);
  auto& scale     = instr->inputs(2);
  auto& save_mean = instr->inputs(3);
  auto& save_var  = instr->inputs(4);

  auto& layout = instr.GetAttrs<std::string>("layout");

  CinnBuilder* builder_ = nullptr;
  std::vector<int> dim  = {};
  float num_element     = 0;
  int channel           = 0;
  int c_dim             = 0;
  if (layout == "NCHW") {
    c_dim       = 1;
    dim         = {0, 2, 3};
    channel     = x->shape[1];
    num_element = x->shape[0] * x->shape[2] * x->shape[3];
  } else {
    c_dim       = 3;
    dim         = {0, 1, 2};
    channel     = x->shape[3];
    num_element = x->shape[0] * x->shape[1] * x->shape[2];
  }

  // grad bias = reduce(dy), shape = [c]
  auto grad_bias = builder->Reduce(dy, ReduceKind::kSum, dim);
  // grad scale = dy * (x - mean)/var, shape = [c]
  auto mean = builder->BroadcastTo(save_mean, x->shape, {c_dim});
  auto var  = builder->BroadcastTo(save_var, x->shape, {c_dim});

  auto diff = builder->Sub(x, mean);
  // grad scale = dy * (diff/var), shape = [c]
  auto grad_scale = builder->Reduce(builder->Mul(dy, builder->Div(diff / var)), ReduceKind::kSum, dim);
  // grad [(x - mean)/var] = dy * scale, shape = [n,c,h,w]
  auto scale_   = builder->BroadcastTo(scale, x->shape, {c_dim});
  auto grad_std = builder->Mul(dy, scale_);

  // grad [diff=(x - mean)] = dstd/var, shape = [n,c,h,w]
  auto grad_diff0 = builder->Div(grad_std, var);
  // grad var = (-1 * grad_std * diff)/(save_var*save_var), shape = [c]
  auto grad_var = builder->Reduce(
      builder->Mul(builder->ConstScalar(-1.0f), builder->Div(builder->Mul(grad_std, diff), builder->Mul(var, var))),
      ReduceKind::kSum,
      dim);
  // grad diff2 = (1.0f / ( 2 * num_element)) * (grad_var / save_var), shape[n,c,h,w]
  auto grad_diff2 = builder->BroadcastTo(
      builder->Mul(builder->ConstScalar(1.0f / num_element), builder->Div(grad_var, save_var)), x->shape, {c_dim});
  // grad diff = (grad_diff2 * 2 * diff), shape = [n,c,h,w]
  auto grad_diff = builder->Add(builder->Mul(grad_diff2, diff), grad_diff0);
  // grad mean, shape = [c]
  auto grad_sum =
      builder->Reduce(builder->Mul(builder->ConstScalar(-1.0f / num_element), grad_diff), ReduceKind::kSum, dim);
  // grad x
  auto grad_x = builder->Add(grad_diff, builder->BroadcastTo(grad_sum, x->shape, {c_dim}));

  // set output
  grad_x.set_id(instr->outputs(0)->id);
  grad_scale.set_id(instr->outputs(1)->id);
  grad_bias.set_id(instr->outputs(2)->id);
}

void conv2d_grad(const Instruction* instr, const DecomposerContext& context) {
  auto& x  = instr->inputs(0);
  auto& w  = instr->inputs(1);
  auto& dy = instr->inputs(2);

  CinnBuilder* cinn_builder = context.builder_;
  // create backward data
  auto dx = cinn_builder->Conv(w,
                               dy,
                               instr.GetAttrs<std::vector>("strides"),
                               instr.GetAttrs<std::vector>("paddings"),
                               instr.GetAttrs<std::vector>("dilations"),
                               instr.GetAttrs<int>("groups"),
                               "backward_data",
                               instr.GetAttrs<std::string>("data_format"),
                               instr.GetAttrs<std::string>("padding_algorithm"));
  dx.set_id(instr.outputs[0]->id);

  // create backward filter
  auto dw = cinn_builder->Conv(x,
                               dy,
                               instr.GetAttrs<std::vector>("strides"),
                               instr.GetAttrs<std::vector>("paddings"),
                               instr.GetAttrs<std::vector>("dilations"),
                               instr.GetAttrs<int>("groups"),
                               "backward_filter",
                               instr.GetAttrs<std::string>("data_format"),
                               instr.GetAttrs<std::string>("padding_algorithm"));

  dw.set_id(instr.outputs[1]->id);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(nn) {
  CINN_DECOMPOSER_REGISTER(batch_norm_train, ::cinn::common::DefaultHostTarget())
      .set_body(cinn::frontend::decomposer::batch_norm_train);
  CINN_DECOMPOSER_REGISTER(batch_norm_train, ::cinn::common::DefaultNVGPUTarget())
      .set_body(cinn::frontend::decomposer::batch_norm_train);

  CINN_DECOMPOSER_REGISTER(batch_norm_grad, ::cinn::common::DefaultHostTarget())
      .set_body(cinn::frontend::decomposer::batch_norm_grad);
  CINN_DECOMPOSER_REGISTER(batch_norm_grad, ::cinn::common::DefaultNVGPUTarget())
      .set_body(cinn::frontend::decomposer::batch_norm_grad);

  CINN_DECOMPOSER_REGISTER(conv2d_grad, ::cinn::common::DefaultHostTarget())
      .set_body(cinn::frontend::decomposer::conv2d_grad);
  CINN_DECOMPOSER_REGISTER(conv2d_grad, ::cinn::common::DefaultNVGPUTarget())
      .set_body(cinn::frontend::decomposer::conv2d_grad);

  return true;
}
