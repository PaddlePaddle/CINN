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

#include "cinn/hlir/framework/accuracy_checker.h"

#include <gtest/gtest.h>

#include <random>
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"

DECLARE_bool(cinn_self_check_accuracy);

namespace cinn {
namespace hlir {
namespace framework {

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

void SetRandomTensor(Tensor tensor, Target target, bool generate_nan) {
  size_t numel = tensor->shape().numel();
  float* dst   = tensor->mutable_data<float>(target);

  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-100.f, 100.f);
  std::vector<float> random_nan_vec(numel);
  for (size_t i = 0; i < numel; i++) {
    float v           = dist(engine);
    random_nan_vec[i] = generate_nan ? sqrt(v) : v;
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst, random_nan_vec.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_nan_vec.begin(), random_nan_vec.end(), dst);
#endif
}

TEST(AccuracyChecker, tensor) {
  Target target = GetTarget();
  Scope scope;
  scope.Var<Tensor>("y");
  auto out = scope.GetTensor("y");
  out->Resize(Shape({16, 16}));
  SetRandomTensor(out, target, true);

  AccuracyChecker checker(target, &scope, {}, {"y"});
  CHECK(checker());
}

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(Target target) {
  Expr m(16);
  Expr n(16);

  lang::Placeholder<float> x("x", {m, n});

  auto y = Compute(
      {m, n}, [=](Expr i, Expr j) { return lang::CallExtern("sqrt", {x(i, j)}); }, "y");

  auto stages = CreateStages({y});
  auto fn     = Lower("fn", stages, {x, y});

  ir::Module::Builder builder("some_module", target);
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

void InstantiateScope(Scope* scope, Target target) {
  for (auto& name : std::vector<std::string>({"x", "y"})) {
    scope->Var<Tensor>(name);
    auto x = scope->GetTensor(name);
    x->Resize(Shape({16, 16}));
    SetRandomTensor(x, target, false);
  }
}

TEST(AccuracyChecker, instruction) {
  Target target = common::DefaultHostTarget();
  Scope scope;
  InstantiateScope(&scope, target);

  Instruction instr(target, &scope, {"x"}, {"y"});
  auto jit     = GetLoweredFunc(target);
  auto fn_addr = jit->Lookup("fn");
  CHECK(fn_addr);

  FLAGS_cinn_self_check_accuracy = true;
  instr.SetLoweredFunc(reinterpret_cast<lower_func_ptr_t>(fn_addr));
  // should call Finalize explicitly before Run
  ASSERT_DEATH(instr.Run(), "");
  instr.Finalize();
  instr.Run();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
