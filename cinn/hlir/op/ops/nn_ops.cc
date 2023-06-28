// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "absl/types/optional.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace hlir {
namespace op {

std::shared_ptr<OpStrategy> StrategyForConv2d(const framework::NodeAttr &attrs,
                                              const std::vector<ir::Tensor> &inputs,
                                              const std::vector<Type> &out_type,
                                              const std::vector<std::vector<int>> &output_shapes,
                                              const Target &target) {
  auto padding  = GetAttr<std::vector<int>>(attrs.attr_store, "padding", {0, 0});
  auto stride   = GetAttr<std::vector<int>>(attrs.attr_store, "stride", {1, 1});
  auto dilation = GetAttr<std::vector<int>>(attrs.attr_store, "dilation", {1, 1});
  CHECK_EQ(padding.size(), 2);
  CHECK_EQ(stride.size(), 2);
  CHECK_EQ(dilation.size(), 2);
  auto data_format = GetAttr<std::string>(attrs.attr_store, "data_format", "NCHW");
  auto groups      = GetAttr<int>(attrs.attr_store, "groups", 1);
  auto conv_type   = GetAttr<std::string>(attrs.attr_store, "conv_type", "forward");
  auto use_mkldnn  = GetAttr<bool>(attrs.attr_store, "use_mkldnn", false);
  auto key         = GetAttr<std::string>(attrs.attr_store, "key", "");

#ifndef CINN_WITH_CUDNN
  CHECK_EQ(conv_type, "forward") << "cudnn is not found, backward_data/backward_filter is not supported!";
#endif

  framework::CINNCompute conv2d_compute([=](lang::Args args, lang::RetValue *ret) {
    std::vector<CINNValue> res;
    CHECK(!args.empty()) << "The input argument of conv2d compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 3U) << "at least 2 input tensors for conv2d compute\n";
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    std::string tensor_name = pack_args[2].operator std::string();
    VLOG(3) << "input shape: " << utils::Join(A.as_tensor_ref()->shape, ", ");
    VLOG(3) << "weight shape: " << utils::Join(B.as_tensor_ref()->shape, ", ");

    std::vector<ir::Tensor> out;
    if (data_format == "NCHW") {
      // A is input: [N, C, H, W], B is filter: [C_out, C_in/group, filter_h, filter_w]
      if (target.arch == Target::Arch::X86) {
        if (groups == 1 && !use_mkldnn) {
          out = pe::Conv2d_NCHW_5D(A.as_tensor_ref(),
                                   B.as_tensor_ref(),
                                   padding[0],
                                   padding[1],
                                   stride[0],
                                   stride[1],
                                   dilation[0],
                                   dilation[1],
                                   key,
                                   tensor_name,
                                   target);
        } else {
#ifdef CINN_WITH_MKLDNN
          out = pe::Conv2d_NCHW_MKLDNN(A.as_tensor_ref(),
                                       B.as_tensor_ref(),
                                       padding[0],
                                       padding[1],
                                       stride[0],
                                       stride[1],
                                       dilation[0],
                                       dilation[1],
                                       tensor_name);
#else
          out = pe::Conv2d_NCHW_5D(A.as_tensor_ref(),
                                   B.as_tensor_ref(),
                                   padding[0],
                                   padding[1],
                                   stride[0],
                                   stride[1],
                                   dilation[0],
                                   dilation[1],
                                   key,
                                   tensor_name);
#endif
        }
      } else {
        if (conv_type == "forward") {
          out = pe::Conv2d_NCHW(A.as_tensor_ref(),
                                B.as_tensor_ref(),
                                padding[0],
                                padding[1],
                                stride[0],
                                stride[1],
                                dilation[0],
                                dilation[1],
                                tensor_name);
        } else {
#ifdef CINN_WITH_CUDNN
          CINN_NOT_IMPLEMENTED
#endif
        }
      }
    } else if (data_format == "NHWC") {
      // A is input: [N, H, W, C], B is filter: [C_out, C_in/group, filter_h, filter_w]
      out = pe::Conv2d_NHWC(A.as_tensor_ref(),
                            B.as_tensor_ref(),
                            padding[0],
                            padding[1],
                            stride[0],
                            stride[1],
                            dilation[0],
                            dilation[1],
                            tensor_name);
    } else {
      LOG(FATAL) << "Only support NCHW and NHWC data layout\n";
    }
    auto stages = CreateStages({A.as_tensor_ref(), B.as_tensor_ref()});

    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(CINNValue(t));
    }
    CHECK(out.size() == 3U || out.size() == 2U || out.size() == 5U || out.size() == 12U)
        << "The output tensor sizes of conv2d op in conv2d op should be 2 or 3 or 5\n";

    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule conv2d_schedule([=](lang::Args args, lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of conv2d schedule is empty! Please check.\n";
      CINNValuePack arg_pack = args[0];
      std::vector<Expr> vec_ast;
      for (int i = 0; i < arg_pack.size(); i++) {
        if (arg_pack[i].is_expr()) {
          Expr temp = arg_pack[i];
          vec_ast.emplace_back(temp);
        }
      }
      CHECK(!vec_ast.empty());
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      ir_sch.MergeExprs();
      if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDNN
        // If conv_type is backward_filter or backward_data, we built a fake op.
        // As runtime use cudnn to compute conv2d, this fake op is not to be called.
        // When cinn support backward_filter/backward_data code gen, this code is to be removed.
        if (conv_type != "forward") {
          CHECK_EQ(vec_ast.size(), 1);
          pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
          std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
          return;
        }
#endif
        int expr_size = vec_ast.size();
        if (expr_size == 2) {
          pe::IRCudaScheduleConv(ir_sch, target);
          VLOG(3) << "After IRCudaScheduleConv, arg_pack[0] is : " << ir_sch.GetModule().GetExprs().at(0);
          std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
          return;
        } else {
          CINN_NOT_IMPLEMENTED
        }
      } else if (target.arch == Target::Arch::X86) {
        CINN_NOT_IMPLEMENTED
      }
      LOG(FATAL) << "This target [" << target << "] is not supported yet.";
    } else {
      CHECK(!args.empty()) << "The input argument of conv2d schedule is empty! Please check.\n";
      CINNValuePack arg_pack = args[0];
      CHECK(arg_pack.size() == 4UL || arg_pack.size() == 3UL || arg_pack.size() == 6UL || arg_pack.size() == 13UL);
      poly::StageMap stages = arg_pack.back();
      if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDNN
        // If conv_type is backward_filter or backward_data, we built a fake op.
        // As runtime use cudnn to compute conv2d, this fake op is not to be called.
        // When cinn support backward_filter/backward_data code gen, this code is to be removed.
        if (conv_type != "forward") {
          Expr out = arg_pack[0];
          pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes.front(), target);
          *ret = CINNValuePack{{CINNValue(out), CINNValue(stages)}};
          return;
        }
#endif
        if (arg_pack.size() == 4UL) {
          Expr Out             = arg_pack[0];
          Expr input_pad       = arg_pack[1];
          Expr weights         = arg_pack[2];
          ir::Tensor out_t     = Out.as_tensor_ref();
          ir::Tensor input_t   = input_pad.as_tensor_ref();
          ir::Tensor weights_t = weights.as_tensor_ref();
          CHECK(Out.as_tensor());
          pe::CudaScheduleConv(stages, input_t, weights_t, out_t, target);
          arg_pack[0] = Expr(out_t);
          arg_pack[1] = Expr(input_t);
          arg_pack[2] = Expr(weights_t);
          *ret        = CINNValuePack{{arg_pack[0], CINNValue(stages)}};
          return;
        } else if (arg_pack.size() == 13UL) {
          Expr wino_weights_dilation          = arg_pack[0];
          Expr wino_input_pad                 = arg_pack[1];
          Expr wino_A                         = arg_pack[2];
          Expr wino_B                         = arg_pack[3];
          Expr wino_G                         = arg_pack[4];
          Expr kernel_pack                    = arg_pack[5];
          Expr input_tile                     = arg_pack[6];
          Expr data_pack                      = arg_pack[7];
          Expr bgemm                          = arg_pack[8];
          Expr inverse                        = arg_pack[9];
          Expr wino_conv                      = arg_pack[10];
          ir::Tensor wino_weights_dilation_t  = wino_weights_dilation.as_tensor_ref();
          ir::Tensor wino_input_pad_t         = wino_input_pad.as_tensor_ref();
          ir::Tensor wino_A_t                 = wino_A.as_tensor_ref();
          ir::Tensor wino_B_t                 = wino_B.as_tensor_ref();
          ir::Tensor wino_G_t                 = wino_G.as_tensor_ref();
          ir::Tensor kernel_pack_t            = kernel_pack.as_tensor_ref();
          ir::Tensor input_tile_t             = input_tile.as_tensor_ref();
          ir::Tensor data_pack_t              = data_pack.as_tensor_ref();
          ir::Tensor bgemm_t                  = bgemm.as_tensor_ref();
          ir::Tensor inverse_t                = inverse.as_tensor_ref();
          ir::Tensor wino_conv_t              = wino_conv.as_tensor_ref();
          std::vector<ir::Tensor> all_tensors = {wino_weights_dilation_t,
                                                 wino_input_pad_t,
                                                 wino_A_t,
                                                 wino_B_t,
                                                 wino_G_t,
                                                 kernel_pack_t,
                                                 input_tile_t,
                                                 data_pack_t,
                                                 bgemm_t,
                                                 inverse_t,
                                                 wino_conv_t};
          hlir::pe::CudaScheduleWinogradConv(stages, all_tensors, target);
          arg_pack[0]  = Expr(all_tensors[0]);
          arg_pack[1]  = Expr(all_tensors[1]);
          arg_pack[2]  = Expr(all_tensors[2]);
          arg_pack[3]  = Expr(all_tensors[3]);
          arg_pack[4]  = Expr(all_tensors[4]);
          arg_pack[5]  = Expr(all_tensors[5]);
          arg_pack[6]  = Expr(all_tensors[6]);
          arg_pack[7]  = Expr(all_tensors[7]);
          arg_pack[8]  = Expr(all_tensors[8]);
          arg_pack[9]  = Expr(all_tensors[9]);
          arg_pack[10] = Expr(all_tensors[10]);
          *ret         = CINNValuePack{{arg_pack[10], arg_pack[5], arg_pack[7], arg_pack[8], CINNValue(stages)}};
          return;
        }
      } else if (target.arch == Target::Arch::X86) {
        if (arg_pack.size() == 6UL) {
          Expr res              = arg_pack[0];
          Expr packed_out       = arg_pack[1];
          Expr weights_dilation = arg_pack[2];
          Expr input_pad        = arg_pack[3];
          Expr data             = arg_pack[4];
          CHECK(res.as_tensor());
          CHECK(packed_out.as_tensor());
          CHECK(input_pad.as_tensor());
          CHECK(weights_dilation.as_tensor());
          CHECK(data.as_tensor());
          std::vector<Expr> kernel_shape = weights_dilation.as_tensor_ref()->shape;
          // kernel_h == 1 && kernel_w == 1
          CHECK_EQ(kernel_shape.size(), 6U) << "kernel_dialtion shape size should be 6";
          bool is_1x1                  = (is_zero(kernel_shape[2] - 1)) && (is_zero(kernel_shape[3] - 1));
          ir::Tensor packed_out_tensor = packed_out.as_tensor_ref();
          bool do_padding              = (padding[0] == 0 && padding[1] == 0) ? false : true;

          if (groups == 1) {
            if (is_1x1) {
              pe::Conv2d_NCHWc_1X1_Schedule_CPU(stages,
                                                res.as_tensor_ref(),
                                                packed_out_tensor,
                                                input_pad.as_tensor_ref(),
                                                weights_dilation.as_tensor_ref(),
                                                data.as_tensor_ref(),
                                                target,
                                                key,
                                                do_padding);
            } else {
              pe::Conv2d_NCHWc_Schedule_CPU(stages,
                                            res.as_tensor_ref(),
                                            packed_out_tensor,
                                            input_pad.as_tensor_ref(),
                                            weights_dilation.as_tensor_ref(),
                                            data.as_tensor_ref(),
                                            target,
                                            key,
                                            do_padding);
            }
            if (do_padding) {
              *ret = CINNValuePack{
                  {CINNValue(res), CINNValue(packed_out_tensor), arg_pack[2], arg_pack[3], CINNValue(stages)}};
            } else {
              *ret = CINNValuePack{{CINNValue(res), CINNValue(packed_out_tensor), arg_pack[2], CINNValue(stages)}};
            }
            return;
          } else {
            // todo: opt group_conv schedule
            VLOG(3) << "use simple group convolution schedule";
            stages[input_pad.as_tensor_ref()]->ComputeInline();
            stages[weights_dilation.as_tensor_ref()]->ComputeInline();
            stages[data.as_tensor_ref()]->ComputeInline();
            *ret = CINNValuePack{{arg_pack[0], CINNValue(packed_out_tensor), CINNValue(stages)}};
          }
          return;
        } else if (arg_pack.size() == 4UL) {
          Expr input_pad = arg_pack[1];
          CHECK(input_pad.as_tensor());
          stages[input_pad.as_tensor_ref()]->ComputeInline();
          Expr weights_dilation = arg_pack[2];
          CHECK(weights_dilation.as_tensor());
          stages[weights_dilation.as_tensor_ref()]->ComputeInline();
          *ret = CINNValuePack{{arg_pack[0], CINNValue(stages)}};
          return;
        }
      }
      *ret = arg_pack;
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  CHECK(out_type.size()) << "Out_type of conv2d op is empty! Please check.";
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(conv2d_compute, conv2d_schedule, "strategy.conv2d.x86", 1);
  } else {
    LOG(FATAL) << "Conv2d op with dtype != float32 is not implemented yet!";
  }
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(nn_ops) { return true; }
