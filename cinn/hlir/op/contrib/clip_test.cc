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

#include "cinn/hlir/op/contrib/clip.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/common/context.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {

TEST(GenerateCode_Cpu, Clip) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  float max_val = 0.8f;
  float min_val = 0.2f;

  lang::Placeholder<float> in("in", {4, 16, 32});
  std::vector<ir::Tensor> res = Clip(in, max_val, min_val, "test_clip");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Clip", stages, res, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Clip_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;
}

TEST(GenerateCode_Cuda, Clip) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultNVGPUTarget();

  float max_val = 0.8f;
  float min_val = 0.2f;

  lang::Placeholder<float> in("in", {4, 16, 32});
  std::vector<ir::Tensor> res = Clip(in, max_val, min_val, "test_clip");

  poly::StageMap stages = poly::CreateStages({res});
  stages[res[0]]->Bind(0, "blockIdx.x");
  stages[res[0]]->Bind(1, "threadIdx.y");
  stages[res[0]]->SetBuffer("global");

  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCuda_Clip", stages, res, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CUDA codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Clip_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn