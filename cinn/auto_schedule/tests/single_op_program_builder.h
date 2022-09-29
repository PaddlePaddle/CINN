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

#pragma once

#include "cinn/auto_schedule/tests/program_case_builder.h"
#include "cinn/frontend/net_builder.h"

namespace cinn {
namespace auto_schedule {

class AddProgramBuilder : public ProgramCaseBuilder {
 public:
  AddProgramBuilder(int M, int N) : M_(M), N_(N) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("add_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, N_}, "X");
    auto y = builder.CreateInput(Float(32), {M_, N_}, "Y");

    auto mul_out = builder.Add(x, y);
    return builder.Build();
  }

 private:
  int M_;
  int N_;
};

class MulProgramBuilder : public ProgramCaseBuilder {
 public:
  MulProgramBuilder(int M, int K, int N) : M_(M), K_(K), N_(N) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("mul_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, K_}, "X");
    auto y = builder.CreateInput(Float(32), {N_, K_}, "Y");

    auto mul_out = builder.Mul(x, y, 1, 1);
    return builder.Build();
  }

 private:
  int M_;
  int K_;
  int N_;
};

class MatmulProgramBuilder : public ProgramCaseBuilder {
 public:
  MatmulProgramBuilder(int M, int K, int N) : M_(M), K_(K), N_(N) {}

  frontend::Program operator()() override {
    frontend::NetBuilder builder("matmul_net_builder");
    auto x = builder.CreateInput(Float(32), {M_, K_}, "X");
    auto y = builder.CreateInput(Float(32), {K_, N_}, "Y");

    auto mul_out = builder.Matmul(x, y);
    return builder.Build();
  }

 private:
  int M_;
  int K_;
  int N_;
};

}  // namespace auto_schedule
}  // namespace cinn
