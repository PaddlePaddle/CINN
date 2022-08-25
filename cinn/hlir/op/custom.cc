// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/common/cas.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

using ArgsFunc = std::function<std::vector<ir::Expr>(
    const framework::NodeAttr &, const std::vector<ir::Tensor> &, const std::vector<std::vector<int>> &)>;

class CustomCallArgsFuncRegistry {
 public:
  static CustomCallArgsFuncRegistry &Global() {
    static CustomCallArgsFuncRegistry instance;
    return instance;
  }

  void Register(const std::string &custom_call, const common::Target &target, ArgsFunc args_func) {
    auto id       = custom_call + target.arch_str();
    func_map_[id] = args_func;
  }

  ArgsFunc Lookup(const std::string &custom_call, const common::Target &target) {
    auto id = custom_call + target.arch_str();
    CHECK(func_map_.count(id));
    return func_map_[id];
  }

 private:
  CustomCallArgsFuncRegistry() {}
  std::unordered_map<std::string, ArgsFunc> func_map_;
};

std::shared_ptr<OpStrategy> StrategyForCustomCall(const framework::NodeAttr &attrs,
                                                  const std::vector<ir::Tensor> &inputs,
                                                  const std::vector<Type> &out_type,
                                                  const std::vector<std::vector<int>> &output_shapes,
                                                  const Target &target) {
  framework::CINNCompute compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK_EQ(args, size(), 1UL);
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 1UL);

    auto attr_store = attrs.attr_store;
    CHECK(attr_store.count("custom_call"));
    std::string custom_call_api = absl::get<std::string>(attr_store.at("custom_call"));
    auto args_func              = CustomCallArgsFuncRegistry::Global().Lookup(custom_call_api, target);

    std::string node_id = pack_args[0] std::string();
    // create call function.
    ir::Var kernel_args(KERNEL_ARGS, type_of<void *>());
    ir::Var kernel_args_num(KERNEL_ARGS_NUM, type_of<int>());

    auto args_list                  = args_func(node->attrs, inputs, output_shapes);
    std::vector<ir::Expr> host_args = {kernel_args, kernel_args_num};
    args.insert(host_args.end(), args_list.begin(), args_list.end());
    auto call_extern_api =
        ir::Call::Make(Void(), custom_call_api, host_args, {}, ir::CallType::Extern, ir::FunctionRef(), 0);
    std::vector<ir::Argument> arguments = {ir::Argument(kernel_args, ir::Argument::IO::kOutput),
                                           ir::Argument(kernel_args_num, ir::Argument::IO::kInput)};
    // if target is nvgpu, add stream.
    if (target_ == common::DefaultNVGPUTarget()) {
      ir::Var kernel_stream(KERNEL_STREAM, type_of<void *>());

      args.push_back(kernel_stream);
      arguments.emplace_back(kernel_stream, ir::Argument::IO::kOutput);
    }
    auto func = ir::_LoweredFunc_::Make("func_" + node_id, arguments, call_extern_api, {});

    *ret = CINNValuePack{{CINNValue(ir::Expr(func))}};
  });

  framework::CINNSchedule schedule([=](lang::Args args, lang::RetValue *ret) {
    }
});

auto strategy = std::make_shared<framework::OpStrategy>();
strategy->AddImpl(compute, schedule, "strategy.custom_call.x86", 1);
return strategy;
}

std::vector<ir::Expr> CustomCallArgsForCublas(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<std::vector<int>> &output_shapes) {
  auto attr_store = attrs.attr_store;
  bool trans_a    = attr_store.count("trans_a") ? absl::get<bool>(attr_store.at("trans_a")) : false;
  bool trans_b    = attr_store.count("trans_b") ? absl::get<bool>(attr_store.at("trans_b")) : false;
  float alpha     = attr_store.count("alpha") ? absl::get<float>(attr_store.at("alpha")) : 1.0f;
  float beta      = attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;
  // m n k
  int m = trans_a ? inputs[0]->shape[1].as_int32() : inputs[0]->shape[0].as_int32();
  int n = trans_b ? inputs[1]->shape[0].as_int32() : inputs[1]->shape[1].as_int32();
  int k = trans_a ? inputs[0]->shape[0].as_int32() : inputs[0]->shape[1].as_int32();
  // func args
  std::vector<ir::Expr> args = {
      ir::Expr(trans_a), ir::Expr(trans_b), ir::Expr(alpha), ir::Expr(beta), ir::Expr(m), ir::Expr(n), ir::Expr(k)};
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnConvForward(const framework::NodeAttr &attrs,
                                                        const std::vector<ir::Tensor> &inputs,
                                                        const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 2UL);
  CHECK_EQ(output_shapes.size(), 1UL);
  auto attr_store = attrs.attr_store;
  float alpha     = attr_store.count("alpha") ? absl::get<float>(attr_store.at("alpha")) : 1.0f;
  float beta      = attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  auto padding  = attr_store.count("padding") ? absl::get<std::vector<int>>(attrs.attr_store.at("padding")) : {0, 0};
  auto stride   = attr_store.count("stride") ? absl::get<std::vector<int>>(attrs.attr_store.at("stride")) : {1, 1};
  auto dilation = attr_store.count("dilation") ? absl::get<std::vector<int>>(attrs.attr_store.at("dilation")) : {1, 1};
  std::string data_format =
      attr_store.count("data_format") ? absl::get<std::string>(attrs.attr_store.at("data_format")) : "NCHW";
  int groups = attrs.attr_store.find("groups") ? absl::get<int>(attrs.attr_store.at("groups")) : 1;

  std::vector<ir::Expr> args = {ir::Expr(alpha), ir::Expr(beta)};
  args.insert(args.end(), inputs[0]->shape.begin(), inputs[0]->shape.end());
  args.insert(args.end(), inputs[1]->shape.begin(), inputs[1]->shape.end());
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  std::transform(output_shapes[0].begin(), output_shapes[0].end(), std::back_inserter(args), [](const int dim) {
    return ir::Expr(dim);
  });
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnConvBackwardData(const framework::NodeAttr &attrs,
                                                             const std::vector<ir::Tensor> &inputs,
                                                             const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 2UL);
  CHECK_EQ(output_shapes.size(), 1UL);
  auto attr_store = attrs.attr_store;
  float alpha     = attr_store.count("alpha") ? absl::get<float>(attr_store.at("alpha")) : 1.0f;
  float beta      = attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  auto padding  = attr_store.count("padding") ? absl::get<std::vector<int>>(attrs.attr_store.at("padding")) : {0, 0};
  auto stride   = attr_store.count("stride") ? absl::get<std::vector<int>>(attrs.attr_store.at("stride")) : {1, 1};
  auto dilation = attr_store.count("dilation") ? absl::get<std::vector<int>>(attrs.attr_store.at("dilation")) : {1, 1};
  std::string data_format =
      attr_store.count("data_format") ? absl::get<std::string>(attrs.attr_store.at("data_format")) : "NCHW";
  int groups = attrs.attr_store.find("groups") ? absl::get<int>(attrs.attr_store.at("groups")) : 1;

  std::vector<ir::Expr> args = {ir::Expr(alpha), ir::Expr(beta)};
  std::transform(output_shapes[0].begin(), output_shapes[0].end(), std::back_inserter(args), [](const int dim) {
    return ir::Expr(dim);
  });
  args.insert(args.end(), inputs[0]->shape.begin(), inputs[0]->shape.end());
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  args.insert(args.end(), inputs[1]->shape.begin(), inputs[1]->shape.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnConvBackwardFilter(const framework::NodeAttr &attrs,
                                                               const std::vector<ir::Tensor> &inputs,
                                                               const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 2UL);
  CHECK_EQ(output_shapes.size(), 1UL);
  auto attr_store = attrs.attr_store;
  float alpha     = attr_store.count("alpha") ? absl::get<float>(attr_store.at("alpha")) : 1.0f;
  float beta      = attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  auto padding  = attr_store.count("padding") ? absl::get<std::vector<int>>(attrs.attr_store.at("padding")) : {0, 0};
  auto stride   = attr_store.count("stride") ? absl::get<std::vector<int>>(attrs.attr_store.at("stride")) : {1, 1};
  auto dilation = attr_store.count("dilation") ? absl::get<std::vector<int>>(attrs.attr_store.at("dilation")) : {1, 1};
  std::string data_format =
      attr_store.count("data_format") ? absl::get<std::string>(attrs.attr_store.at("data_format")) : "NCHW";
  int groups = attrs.attr_store.find("groups") ? absl::get<int>(attrs.attr_store.at("groups")) : 1;

  std::vector<ir::Expr> args = {ir::Expr(alpha), ir::Expr(beta)};
  args.insert(args.end(), inputs[0]->shape.begin(), inputs[0]->shape.end());
  std::transform(output_shapes[0].begin(), output_shapes[0].end(), std::back_inserter(args), [](const int dim) {
    return ir::Expr(dim);
  });
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  args.insert(args.end(), inputs[1]->shape.begin(), inputs[1]->shape.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnPoolForward(const framework::NodeAttr &attrs,
                                                        const std::vector<ir::Tensor> &inputs,
                                                        const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 1UL);
  CHECK_EQ(output_shapes.size(), 1UL);
  auto attr_store = attrs.attr_store;

  auto kernel =
      attr_store.count("kernel_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("kernel_size")) : {1, 1};
  auto padding =
      attr_store.count("padding_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("padding_size")) : {0, 0};
  auto stride =
      attr_store.count("stride_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("stride_size")) : {0, 0};
  int pool_type = attr_store.count("pool_type")
                      ? absl::get<std::vector<int>>(attrs.attr_store.at("pool_type")) == "max" ? 0 : 1
                      : 1;
  std::string data_format =
      attr_store.count("data_format") ? absl::get<std::string>(attrs.attr_store.at("data_format")) : "NCHW";

  std::vector<ir::Expr> args = {pool_type};
  args.insert(args.end(), inputs[0]->shape.begin(), inputs[0]->shape.end());
  args.push_back(ir::Expr(kernel[0]));
  args.push_back(ir::Expr(kernel[1]));
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  std::transform(output_shapes[0].begin(), output_shapes[0].end(), std::back_inserter(args), [](const int dim) {
    return ir::Expr(dim);
  });
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnPoolBackward(const framework::NodeAttr &attrs,
                                                         const std::vector<ir::Tensor> &inputs,
                                                         const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 2UL);
  CHECK_EQ(output_shapes.size(), 1UL);
  auto attr_store = attrs.attr_store;

  auto kernel =
      attr_store.count("kernel_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("kernel_size")) : {1, 1};
  auto padding =
      attr_store.count("padding_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("padding_size")) : {0, 0};
  auto stride =
      attr_store.count("stride_size") ? absl::get<std::vector<int>>(attrs.attr_store.at("stride_size")) : {0, 0};
  int pool_type = attr_store.count("pool_type")
                      ? absl::get<std::vector<int>>(attrs.attr_store.at("pool_type")) == "max" ? 0 : 1
                      : 1;
  std::string data_format =
      attr_store.count("data_format") ? absl::get<std::string>(attrs.attr_store.at("data_format")) : "NCHW";

  std::vector<ir::Expr> args = {pool_type};
  std::transform(output_shapes[0].begin(), output_shapes[0].end(), std::back_inserter(args), [](const int dim) {
    return ir::Expr(dim);
  });
  args.push_back(ir::Expr(kernel[0]));
  args.push_back(ir::Expr(kernel[1]));
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.insert(args.end(), inputs[0]->shape.begin(), inputs[0]->shape.end());

  return args;
}

bool RegisteryCustomCallArgsFunc() {
#ifdef CINN_WITH_CUDA
  CustomCallArgsFuncRegistry::Global().Register("cinn_call_cublas", CustomCallArgsForCublas);
#endif

#ifdef CINN_WITH_CUDNN
  CustomCallArgsFuncRegistry::Global().Register("cinn_call_cudnn_conv2d_forward", CustomCallArgsForCudnnConvForward);
  CustomCallArgsFuncRegistry::Global().Register("cinn_call_cudnn_conv2d_backward_data",
                                                CustomCallArgsForCudnnConvBackwardData);
  CustomCallArgsFuncRegistry::Global().Register("cinn_gpu_cudnn_conv2d_backward_filter",
                                                CustomCallArgsForCudnnConvBackwardFilter);
  CustomCallArgsFuncRegistry::Global().Register("cinn_call_cudnn_pool2d_forward", CustomCallArgsForCudnnPoolForward);
  CustomCallArgsFuncRegistry::Global().Register("cinn_call_cudnn_pool2d_backward", CustomCallArgsForCudnnPoolBackward);
#endif

#ifdef CINN_WITH_MKLDNN

#endif

#ifdef CINN_WITH_MKL_CBLAS

#endif
  return true;
}

static bool registry_custom_call_list_func = RegisteryCustomCallArgsFunc();
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(custom_call_op) {
  CINN_REGISTER_OP(custom_call)
      .describe("This operator implements the call of extern api!")
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForCustomCall)
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kOpaque);

  return true;
}
