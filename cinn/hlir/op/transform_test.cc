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

#include <cuda_runtime.h>
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
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/cuda/cuda_module.h"

namespace cinn {
namespace hlir {
namespace framework {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;
using runtime::cuda::CUDAModule;

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
  attrs.attr_store["strides"] = std::vector<int>{32, 32};

  std::vector<Type> out_type{Float(32)};
  std::vector<int> output_shape = {64, 64};
  std::vector<ir::Tensor> inputs{input.tensor(), assign.tensor()};

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
#else
  auto target = common::DefaultHostTarget();
#endif
  auto impl = OpStrategy::SelectImpl(strategy(attrs, inputs, out_type, {output_shape}, target));

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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
