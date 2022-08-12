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
#include "cinn/backends/cuda_util.h"
#include "cinn/common/context.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;

void GenOneHotCode(
    std::vector<ir::Expr> indices_shape, int axis, int depth, const std::string& dtype, const std::string& device) {
  common::Context::Global().ResetNameId();

  common::Target target;
  if (device == "cuda") {
    target = common::DefaultNVGPUTarget();
  } else if (device == "cpu") {
    target = common::DefaultHostTarget();
  } else {
    LOG(FATAL) << "Unknown device: " << device;
  }

  lang::Placeholder<float> indices("indices", indices_shape);
  lang::Placeholder<float> on_value("on_value", {Expr(1)});
  lang::Placeholder<float> off_value("off_value", {Expr(1)});
  ir::Tensor res = OneHot(indices, on_value, off_value, depth, axis, dtype, "test_one_hot");

  poly::StageMap stages = poly::CreateStages({res});
  if (device == "cuda") {
    stages[res]->Bind(0, "blockIdx.x");
    stages[res]->Bind(1, "threadIdx.y");
    stages[res]->SetBuffer("global");
  }
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_OneHot", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before " << device << " codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("OneHot_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  if (device == "cpu") {
    backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
    VLOG(6) << "Cpu Codegen result:";
    VLOG(6) << code << std::endl;
  }
}

TEST(OneHotOpTest, OneHotTest_Base) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Axis_1) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = 1;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 9, 3, 4, 5};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Axis_Ndim) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = 4;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Depth) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 20;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 20};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Dtype) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 9;
  const std::string dtype             = "int32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Int(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, "cuda");
#endif
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn