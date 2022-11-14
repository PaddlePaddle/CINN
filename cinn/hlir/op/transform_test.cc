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

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/flags.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace framework {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

TEST(SliceAssign, SliceAssign_Op) {
  // code gen
  auto slice_assign = Operator::Get("slice_assign");
  auto strategy     = Operator::GetAttrs<StrategyFunction>("CINNStrategy")[slice_assign];

  int m = 64;
  int n = 32;

  Placeholder<float> input("input", {ir::Expr(m), ir::Expr(m)});
  Placeholder<float> assign("assign", {ir::Expr(n), ir::Expr(n)});

  // set attrs
  NodeAttr attrs;
  attrs.attr_store["axis"]    = std::vector<int>{0, 1};
  attrs.attr_store["starts"]  = std::vector<int>{16, 16};
  attrs.attr_store["ends"]    = std::vector<int>{32, 32};
  attrs.attr_store["strides"] = std::vector<int>{1, 1};

  std::vector<Type> out_type{Float(32)};
  std::vector<int> output_shape = {64, 64};
  std::vector<ir::Tensor> inputs{input.tensor(), assign.tensor()};

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
#else
  auto target = common::DefaultHostTarget();
#endif
  auto impl = OpStrategy::SelectImpl(strategy(attrs, inputs, out_type, {output_shape}, target));

  std::string func_name = "slice_assign";

  if (FLAGS_cinn_ir_schedule) {
    std::string out_name             = "output";
    common::CINNValuePack cinn_input = common::CINNValuePack{
        {common::CINNValue(input.tensor()), common::CINNValue(assign.tensor()), common::CINNValue(out_name)}};
    std::vector<std::string> input_output_names{"input", "assign", out_name};

    auto funcs = framework::GetFuncFromImpl(impl, cinn_input, inputs, input_output_names, func_name, target);

    for (auto func : funcs) {
      LOG(INFO) << "Test Operator_BroadcastTo's Strategy, func is :\n" << func;
    }
  } else {
    common::CINNValuePack cinn_input =
        common::CINNValuePack{{common::CINNValue(input.tensor()), common::CINNValue(assign.tensor())}};
    common::CINNValuePack rets = impl->fcompute(cinn_input);
    rets                       = impl->fschedule(rets);

    // the last element is a StageMap
    for (int i = 0; i < rets->size() - 1; i++) {
      Expr temp = rets[i];
      if (!temp.as_tensor_ref()->buffer.defined()) {
        inputs.push_back(temp.as_tensor_ref());
      }
    }

    auto func = lang::LowerVec("slice_assign", rets.back(), inputs, {}, {}, nullptr, target);
    for (auto& f : func) {
      LOG(INFO) << "Test Strategy Codegen:\n" << f;
    }
  }
}

/*
TEST(Gather, Gather_Op) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  ir::Expr n(4);
  ir::Expr h_in1(28);
  ir::Expr h_in2(14);

  lang::Placeholder<float> in1("in1", {n, h_in1});
  lang::Placeholder<int32_t> in2("in2", {n, h_in2});
  ir::Tensor res = pe::Gather(in1, in2, 1, "test_gather_out");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Gather", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Gather_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void TestGenerateCodeCpu_Gather(void* _args, int32_t num_args)
{
  cinn_buffer_t* _test_gather_out = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _in1 = cinn_buffer_t::new_((cinn_device_kind_t)(0), cinn_float32_t(), { 4, 28 });
  cinn_buffer_t* _in2 = cinn_buffer_t::new_((cinn_device_kind_t)(0), cinn_int32_t(), { 4, 14 });
  cinn_buffer_malloc((void*)(0), _test_gather_out);
  cinn_buffer_malloc((void*)(0), _in1);
  cinn_buffer_malloc((void*)(0), _in2);
  const float* in1 = ((const float*)(_in1->memory));
  const int32_t* in2 = ((const int32_t*)(_in2->memory));
  float* test_gather_out = ((float*)(_test_gather_out->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 14; j += 1) {
      test_gather_out[((14 * i) + j)] = in1[((28 * i) + in2[((14 * i) + j)])];
    };
  };
  cinn_buffer_free((void*)(0), _in1);
  cinn_buffer_free((void*)(0), _in2);
  cinn_buffer_free((void*)(0), _test_gather_out);
}
  )ROC";
  CHECK_EQ(utils::Trim(code), utils::Trim(target_source));
}
*/

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
