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
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/common/cuda_test_helper.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/use_extern_funcs.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace backends {

TEST(CodeGenCUDA, Module_output) {
  Expr M(100);
  Expr N(200);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");

  CodeGenCUDA_Dev codegen(target);

  auto func = Lower("elementwise_mul", stages, {A, B, C});

  Module::Builder builder("module", target);
  builder.AddFunction(func);

  Outputs outputs;
  outputs = outputs.cuda_source("_generated1.cu");
  codegen.Compile(builder.Build(), outputs);
}

}  // namespace backends
}  // namespace cinn
