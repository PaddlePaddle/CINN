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

#include "cinn/hlir/op/contrib/one_hot.h"

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

TEST(GenerateCode_Cpu, Pool2dGrad_Avg) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  int depth                           = 5;
  int axis                            = -1;
  const std::string dtype             = "float32";
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};

  lang::Placeholder<int> indices("indices", indices_shape);
  lang::Placeholder<float> on_value("on_value", {Expr(1)});
  lang::Placeholder<float> off_value("off_value", {Expr(1)});
  ir::Tensor res = OneHot(indices, on_value, off_value, depth, axis, dtype, "test_one_hot");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_OneHot", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("OneHot_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;
}

// TEST(GenerateCode_Cuda, Pool2dGrad_Avg) {
//   common::Context::Global().ResetNameId();

//   common::Target target = common::DefaultNVGPUTarget();

//   std::vector<int> kernel_size = {5, 5};
//   std::vector<int> strides     = {5, 5};
//   std::vector<int> paddings    = {1, 1, 1, 1};
//   ir::Expr n(4);
//   ir::Expr c(3);
//   ir::Expr in_h(28);
//   ir::Expr in_w(28);
//   ir::Expr out_h(6);
//   ir::Expr out_w(6);

//   lang::Placeholder<float> in("in", {n, c, in_h, in_w});
//   lang::Placeholder<float> out("out", {n, c, out_h, out_w});
//   lang::Placeholder<float> out_grad("out_grad", {n, c, out_h, out_w});
//   std::vector<ir::Tensor> res = Pool2dGrad(
//       in, out, out_grad, kernel_size, strides, paddings, "avg", false, false, false, "NCHW", "test_pool2d_in_grad");

//   poly::StageMap stages = poly::CreateStages({res});
//   stages[res[0]]->Bind(0, "blockIdx.x");
//   stages[res[0]]->Bind(1, "threadIdx.y");
//   stages[res[0]]->SetBuffer("global");

//   std::vector<ir::LoweredFunc> funcs =
//       lang::LowerVec("TestGenerateCodeCuda_Pool2dGrad_Avg", stages, res, {}, {}, nullptr, target, true);

//   VLOG(6) << "Expr before CUDA codegen:";
//   VLOG(6) << funcs[0]->body;

//   ir::Module::Builder builder("Pool2dGrad_Avg_Module", target);
//   for (auto& f : funcs) {
//     builder.AddFunction(f);
//   }

//   /*
//   backends::CodeGenCUDA_Dev codegen(target);
//   std::string code = codegen.Compile(builder.Build());
//   VLOG(6) << "Cuda Codegen result:";
//   VLOG(6) << code << std::endl;
//   */
// }

}  // namespace op
}  // namespace hlir
}  // namespace cinn