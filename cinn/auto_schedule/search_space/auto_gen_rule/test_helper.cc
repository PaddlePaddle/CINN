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

#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory.h>
#include <stdlib.h>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/cinn.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/optim/transform_gpu_forloop.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;

ir::IRSchedule TestAutoGenRuleBase::Initialize(const std::string& func_name,
                                               const std::vector<std::vector<int>>& input_shapes,
                                               const std::vector<std::vector<int>>& output_shapes) {
  func_name_     = func_name;
  input_shapes_  = input_shapes;
  output_shapes_ = output_shapes;
#ifdef CINN_WITH_CUDA
  target_ = common::DefaultNVGPUTarget();
#else
  target_ = common::DefaultHostTarget();
#endif
  backend_compier_ = backends::Compiler::Create(target_);

  lowered_funcs_ = GenLoweredFuncs();
  CHECK_GE(tensor_args_.size(), output_shapes.size())
      << "size of tensor_args_ should be as least equal to the number of output variables";
  CHECK_GE(stages_->size(), output_shapes.size())
      << "size of stages_ should be as least equal the number of output variables";
  CHECK(!lowered_funcs_.empty()) << "lowered_funcs_ is empty";

  std::vector<Expr> bodys;
  for (auto&& func : lowered_funcs_) {
    bodys.emplace_back(func->body);
  }
  return ir::IRSchedule(ir::ModuleExpr({std::move(bodys)}));
}

std::vector<ir::LoweredFunc> TestAutoGenRuleBase::Lower2DMatmul(const int mi, const int ki, const int ni) {
  ir::Expr M(mi);
  ir::Expr N(ni);
  ir::Expr K(ki);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  this->stages_      = CreateStages({C});
  this->tensor_args_ = {A, B, C};
  return lang::LowerVec(func_name_, this->stages_, {A, B, C}, {}, {}, nullptr, this->target_, true);
}

ir::Module TestAutoGenRuleBase::BuildIRModule(const std::vector<ir::Expr>& updated_bodys) {
  CHECK_EQ(lowered_funcs_.size(), updated_bodys.size()) << "associated exprs size not equal";

  ir::Module::Builder builder("test_bulder", this->target_);
  for (int i = 0; i < lowered_funcs_.size(); ++i) {
    ir::Expr func_body              = updated_bodys.at(i);
    const ir::LoweredFunc& ori_func = lowered_funcs_.at(i);

    auto temp_buffers = lang::GetTempBuffers(this->tensor_args_, this->stages_, func_body);
    auto new_func     = ir::_LoweredFunc_::Make(ori_func->name, ori_func->args, func_body, temp_buffers);

#ifdef CINN_WITH_CUDA
    optim::OptimizeExprGPU(&func_body);
    new_func->PrepareCudaAxisInfoFromBody();
#endif
    new_func = optim::Optimize(Expr(new_func), this->target_, false).as_lowered_func_ref();
    new_func->PrepareBufferCastExprs(/*with_expr_gen_tensor = */ false);

    builder.AddFunction(new_func);
  }

  return builder.Build();
}

std::string TestAutoGenRuleBase::GenSourceCode(const ir::Module& ir_module) {
#ifdef CINN_WITH_CUDA
  backends::CodeGenCUDA_Dev codegen(this->target_);
#else
  backends::CodeGenCX86 codegen(this->target_, CodeGenCX86::Feature::AVX512);
#endif
  codegen.SetInlineBuiltinCodes(false);
  return codegen.Compile(ir_module, CodeGenC::OutputKind::CImpl);
}

void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void expected_func_matmul(const std::vector<float*>& inputs,
                          const std::vector<float*>& outputs,
                          const std::vector<std::vector<int>>& input_shapes,
                          const std::vector<std::vector<int>>& output_shapes) {
  CHECK_EQ(inputs.size(), 2) << "The number of inputs for matmul must be 2.";
  CHECK_EQ(input_shapes.size(), 2) << "The number of inputs for matmul must be 2.";
  CHECK_EQ(outputs.size(), 1) << "The number of outputs for matmul must be 1.";
  CHECK_EQ(output_shapes.size(), 1) << "The number of outputs for matmul must be 1.";
  CHECK_EQ(input_shapes[0].size(), 2) << "The dimension of the first input must be 2";
  CHECK_EQ(input_shapes[1].size(), 2) << "The dimension of the second input must be 2";
  CHECK_EQ(output_shapes[0].size(), 2) << "The dimension of the output must be 2";
  CHECK_EQ(input_shapes[0][1], input_shapes[1][0])
      << "The second dimension of the first matrix and the first dimension of the second matrix must be equal";
  CHECK_EQ(input_shapes[0][0], output_shapes[0][0])
      << "The first dimension of the first input matrix and the first dimension of the output matrix must be equal";
  CHECK_EQ(input_shapes[1][1], output_shapes[0][1])
      << "The second dimension of the second input matrix and the second dimension of the output matrix must be equal";
  int M = input_shapes[0][0];
  int N = input_shapes[1][1];
  int K = input_shapes[0][1];
  naive_matmul(inputs[0], inputs[1], outputs[0], M, N, K);
}

void MemoryCopy(const float* src, float* dst, int numel, std::string type) {
#ifdef CINN_WITH_CUDA
  if (type == "DeviceToHost") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyDeviceToHost);
    return;
  } else if (type == "HostToDevice") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
#endif
  if (type == "HostToHost") {
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = src[i];
    }
  } else {
    LOG(FATAL) << "Unknown memory copy type";
  }
}

void AddDataToScope(
    Scope* scope, const common::Target& target, float* data_ptr, std::string name, const std::vector<int>& shape) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = absl::get<Tensor>(*var);
  CHECK(shape.size()) << "The size of shape can not be 0.";
  Shape cinn_shape(shape);
  tensor->Resize(cinn_shape);
  auto* tgt_data_ptr = tensor->mutable_data<float>(target);
  std::string mem_cpy_type;
#ifdef CINN_WITH_CUDA
  mem_cpy_type = "HostToDevice";
#else
  mem_cpy_type = "HostToHost";
#endif
  MemoryCopy(data_ptr, tgt_data_ptr, cinn_shape.numel(), mem_cpy_type);
}

using expected_func_type = void (*)(const std::vector<float*>&,
                                    const std::vector<float*>&,
                                    const std::vector<std::vector<int>>&,
                                    const std::vector<std::vector<int>>&);
using test_func_type     = void (*)(void**, int32_t);

void CheckResult(test_func_type test_func,
                 expected_func_type expected_func,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 const std::vector<std::vector<int>>& input_shapes,
                 const std::vector<std::vector<int>>& output_shapes,
                 const common::Target& target) {
  CHECK(input_names.size()) << "The number of inputs must be greater than 0.";
  CHECK(output_names.size()) << "The number of outputs must be greater than 0.";
  CHECK_EQ(input_names.size(), input_shapes.size()) << "The quantity of input_names and input_shapes must be equal.";
  CHECK_EQ(output_names.size(), output_shapes.size())
      << "The quantity of output_names and output_shapes must be equal.";

  // Initialize data
  std::vector<float*> input_data_ptrs(input_names.size());
  for (int i = 0; i < input_shapes.size(); ++i) {
    int input_data_numel =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1, [](int a, int b) { return a * b; });
    input_data_ptrs[i] = reinterpret_cast<float*>(malloc(input_data_numel * sizeof(float)));
    for (int j = 0; j < input_data_numel; ++j) {
      input_data_ptrs[i][j] = (rand() * 1.f) / RAND_MAX;
    }
  }
  std::vector<float*> output_data_ptrs(output_names.size());
  std::vector<float*> expected_output_data_ptrs(output_names.size());
  std::vector<int> output_data_numels(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    output_data_numels[i] =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1, [](int a, int b) { return a * b; });
    ;
    output_data_ptrs[i] = reinterpret_cast<float*>(malloc(output_data_numels[i] * sizeof(float)));
    memset(output_data_ptrs[i], 0, output_data_numels[i] * sizeof(float));
    expected_output_data_ptrs[i] = reinterpret_cast<float*>(malloc(output_data_numels[i] * sizeof(float)));
    memset(expected_output_data_ptrs[i], 0, output_data_numels[i] * sizeof(float));
  }

  // Initialize scope
  Scope scope;
  // Initialize input data in scope.
  for (int i = 0; i < input_names.size(); ++i) {
    AddDataToScope(&scope, target, input_data_ptrs[i], input_names[i], input_shapes[i]);
  }
  // Initialize output data in scope.
  for (int i = 0; i < output_names.size(); ++i) {
    AddDataToScope(&scope, target, output_data_ptrs[i], output_names[i], output_shapes[i]);
  }

  // Create Instruction and run
  Instruction instr(target, &scope, input_names, output_names);
  CHECK(test_func) << "The test_func can not be nullptr.";
  instr.SetLoweredFunc(reinterpret_cast<void*>(test_func));
  // should call Finalize explicitly before Run
  instr.Finalize();
  instr.Run();

  // Get data
  for (int i = 0; i < output_names.size(); ++i) {
    const float* result_ptr = scope.GetTensor(output_names[i])->data<float>();
    std::string mem_cpy_type;
#ifdef CINN_WITH_CUDA
    mem_cpy_type = "DeviceToHost";
#else
    mem_cpy_type = "HostToHost";
#endif
    MemoryCopy(result_ptr, output_data_ptrs[i], output_data_numels[i], mem_cpy_type);
  }

  // Calculate expected result
  expected_func(input_data_ptrs, expected_output_data_ptrs, input_shapes, output_shapes);

  // Check result
  for (int i = 0; i < output_shapes.size(); ++i) {
    for (int j = 0; j < output_data_numels[i]; ++j) {
      ASSERT_NEAR(output_data_ptrs[i][j], expected_output_data_ptrs[i][j], 1e-5);
    }
  }

  // Free memory
  for (auto ptr : input_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : output_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : expected_output_data_ptrs) {
    free(ptr);
  }
}

}  // namespace auto_schedule
}  // namespace cinn
