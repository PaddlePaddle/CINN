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

#include <absl/types/any.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <string>
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
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/packed_func.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "cinn/runtime/use_extern_funcs.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<float> test_mul(const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
  std::vector<float> C_target(M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        //#ifdef CINN_WITH_CUDNN
        C_target[i * N + j] += A[i * K + k] * B[k * N + j];
        /* #else
                C_target[i * N + j] += A[i * K + k] * B[j * N + k];
        #endif */
      }
    }
  }
  return C_target;
}

Tensor GetTensor(const std::shared_ptr<Scope>& scope, const std::string& name) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = absl::get<Tensor>(*var);
  return tensor;
}

void CudaSetRandData(const Tensor& tensor, const Target& target) {
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (float& v : host_memory) {
    v = (rand() * 1.f) / RAND_MAX;  // All random data
  }
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_memory.data(),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyHostToDevice));
}

std::vector<float> CudaGetData(const Tensor& tensor, const Target& target) {
  auto* A_data = tensor->mutable_data<float>(target);
  std::vector<float> host_data(tensor->shape().numel(), 0);

  CUDA_CALL(cudaMemcpy(host_data.data(),
                       reinterpret_cast<void*>(A_data),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return host_data;
}

TEST(GraphCompiler, RunModel) {
  using attr_t = hlir::framework::AttrType;
  frontend::Program prog;
  Expr M(30);
  Expr K(30);
  Expr N(30);
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t   = Float(32);
  a->shape = {M.as_int32(), K.as_int32(), 1, 1};
  b->shape = {N.as_int32(), K.as_int32()};
  a->type  = t;
  b->type  = t;
  auto c   = prog.mul(a, b);
  auto d   = prog.add(c, b);  // N must = K
  auto e   = prog.relu(d);
  absl::flat_hash_map<std::string, attr_t> attr_store;
  attr_store["scale"] = 2.0f;
  attr_store["bias"]  = 0.5f;
  auto o              = prog.scale(e, attr_store);
  ASSERT_EQ(prog.size(), 4UL);
  Target target(Target::OS::Linux, Target::Arch::NVGPU, Target::Bit::k64, {});
  auto g = std::make_shared<Graph>(prog, target);
  ApplyPass(g.get(), "InferShape");

  auto scope = BuildScope(target, g);

  GraphCompiler gc(target, scope, g);
  std::unique_ptr<Program> program = gc.Build();

  auto A = GetTensor(scope, "A");
  auto B = GetTensor(scope, "B");
  CudaSetRandData(A, target);
  CudaSetRandData(B, target);

  program->Execute();
  auto host_data1 = CudaGetData(A, target);
  auto host_data2 = CudaGetData(B, target);
  auto Out        = GetTensor(scope, o->id);
  auto host_data3 = CudaGetData(Out, target);

  auto target_mul = test_mul(host_data1, host_data2, M.as_int32(), K.as_int32(), N.as_int32());
  for (int i = 0; i < Out->shape().numel(); i++) {
    LOG_FIRST_N(INFO, 10) << "cinn_data[" << i << "]: " << 2 * (host_data2[i] + target_mul[i]) + 0.5
                          << " v.s. target_data[" << i << "]: " << host_data3[i];
    EXPECT_NEAR(host_data3[i], 2 * (host_data2[i] + target_mul[i]) + 0.5, 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn
