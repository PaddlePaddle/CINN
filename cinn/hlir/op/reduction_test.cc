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

void CpuReduceSum(const float* x,
                  std::vector<float>* sum0,
                  std::vector<float>* sum1,
                  const int n,
                  const int c,
                  const int h,
                  const int w) {
  memset(sum0->data(), 0, sizeof(float) * c * w);
  memset(sum1->data(), 0, sizeof(float) * c);
  for (int idx = 0; idx < n; ++idx) {
    for (int idy = 0; idy < c; ++idy) {
      for (int idz = 0; idz < h; ++idz) {
        for (int ida = 0; ida < w; ++ida) {
          sum0->at(idy * w + ida) += x[idx * c * h * w + idy * h * w + idz * w + ida];
          sum1->at(idy) += x[idx * c * h * w + idy * h * w + idz * w + ida];
        }
      }
    }
  }
}

std::pair<ir::Module, std::string> GenHostAndPtx(const std::vector<int>& shape,
                                                 const std::vector<int>& dim,
                                                 const std::string func_name = "reduce_sum") {
  auto reduce_sum = Operator::Get("reduce_sum");
  auto strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy")[reduce_sum];

  std::vector<Expr> shape_as_expr;
  for (auto value : shape) {
    shape_as_expr.emplace_back(value);
  }
  Placeholder<float> X("X", shape_as_expr);

  NodeAttr attrs;
  attrs.attr_store["dim"] = dim;
  std::vector<ir::Tensor> inputs{X.tensor()};
  std::vector<Type> out_type{Float(32)};

  std::vector<int> output_shape;
  for (auto value : shape) {
    if (std::find(dim.begin(), dim.end(), value) == dim.end()) {
      output_shape.push_back(value);
    }
  }

  auto target = common::DefaultNVGPUTarget();
  auto impl   = OpStrategy::SelectImpl(strategy(attrs, inputs, out_type, {output_shape}, target));

  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(X)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);

  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = lang::LowerVec(func_name, rets.back(), inputs, {}, {}, nullptr, target);
  for (auto& f : func) {
    LOG(INFO) << "Test Strategy Codegen:\n" << f;
  }

  Module::Builder builder("reduce_sum_0", target);
  for (auto& f : func) {
    builder.AddFunction(f);
  }

  auto module                    = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module              = std::get<0>(host_module_device_module);
  auto& device_module            = std::get<1>(host_module_device_module);
  for (auto& func : host_module.functions()) {
    LOG(INFO) << "host:\n" << func;
  }
  for (auto& func : device_module.functions()) {
    LOG(INFO) << "device:\n" << func;
  }

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  using runtime::cuda::CUDAModule;
  backends::NVRTC_Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  return std::pair<ir::Module, std::string>(host_module, ptx);
}

TEST(Operator, Operator_Reduction_Sum0) {
  int n = 128, c = 128, h = 32, w = 32;
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim   = {0, 2, 3};
  auto module_ptx        = GenHostAndPtx(shape, dim);

  CUDA_CALL(cudaSetDevice(0));
  CUDAModule cuda_module(module_ptx.second, CUDAModule::Kind::PTX);
  void* reduce_sum_kernel = cuda_module.GetFunction(0, "reduce_sum");
  CHECK(reduce_sum_kernel);
  void* reduce_sum_1_kernel = cuda_module.GetFunction(0, "reduce_sum_1");
  CHECK(reduce_sum_1_kernel);

  void* stream = nullptr;
  backends::RuntimeSymbolRegistry::Global().RegisterFn("reduce_sum_kernel_ptr_",
                                                       reinterpret_cast<void*>(&reduce_sum_kernel));
  backends::RuntimeSymbolRegistry::Global().RegisterFn("reduce_sum_1_kernel_ptr_",
                                                       reinterpret_cast<void*>(&reduce_sum_1_kernel));
  backends::RuntimeSymbolRegistry::Global().RegisterVar("reduce_sum_kernel_stream_ptr_", stream);
  backends::RuntimeSymbolRegistry::Global().RegisterVar("reduce_sum_1_kernel_stream_ptr_", stream);

  auto jit = backends::SimpleJIT::Create();
  jit->Link<backends::CodeGenCUDA_Host>(module_ptx.first);

  auto fn_reduce_sum = jit->Lookup("reduce_sum");
  CHECK(fn_reduce_sum);
  auto fn_reduce_sum_1 = jit->Lookup("reduce_sum_1");
  CHECK(fn_reduce_sum_1);

  auto func_0 = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_reduce_sum);
  auto func_1 = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_reduce_sum_1);

  srand(time(NULL));
  auto buffer_x = common::BufferBuilder(Float(32), {n, c, h, w}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {c, w}).set_random().Build();
  auto buffer_z = common::BufferBuilder(Float(32), {c}).set_random().Build();

  void *dev_x = nullptr, *dev_y = nullptr, *dev_z = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));
  CUDA_CALL(cudaMalloc(&dev_z, buffer_z->memory_size));

  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _y;
  cinn_buffer_t _z;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _y.memory = static_cast<uint8_t*>(dev_y);
  _z.memory = static_cast<uint8_t*>(dev_z);

  _x.memory_size = buffer_x->memory_size;
  _y.memory_size = buffer_y->memory_size;
  _z.memory_size = buffer_z->memory_size;

  cinn_pod_value_t x_arg(&_x), y_arg(&_y), z_arg(&_z);
  cinn_pod_value_t args0[] = {x_arg, y_arg};
  cinn_pod_value_t args1[] = {x_arg, y_arg, z_arg};

  func_0(args0, 2);
  func_1(args1, 3);

  CUDA_CALL(cudaMemcpy(buffer_y->memory, dev_y, buffer_y->memory_size, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(buffer_z->memory, dev_z, buffer_z->memory_size, cudaMemcpyDeviceToHost));

  std::vector<float> sum0(c * w);
  std::vector<float> sum1(c);
  CpuReduceSum(reinterpret_cast<float*>(buffer_x->memory), &sum0, &sum1, n, c, h, w);

  std::vector<std::pair<std::vector<float>, float*>> results = {{sum0, reinterpret_cast<float*>(buffer_y->memory)},
                                                                {sum1, reinterpret_cast<float*>(buffer_z->memory)}};
  for (auto& res : results) {
    for (int idx = 0; idx < res.first.size(); ++idx) {
      ASSERT_LT(abs(res.first[idx] - res.second[idx]) / res.first[idx], 1e-4);
    }
  }

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_y));
  CUDA_CALL(cudaFree(dev_z));
}

TEST(Operator, Operator_Reduction_Sum1) {
  int n = 32, c = 32, h = 128, w = 128;
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim   = {0, 1};
  auto module_ptx        = GenHostAndPtx(shape, dim, "reduce_sum_test_1");

  CUDA_CALL(cudaSetDevice(0));
  CUDAModule cuda_module(module_ptx.second, CUDAModule::Kind::PTX);
  void* reduce_sum_test_1_kernel = cuda_module.GetFunction(0, "reduce_sum_test_1");
  CHECK(reduce_sum_test_1_kernel);

  void* stream = nullptr;
  backends::RuntimeSymbolRegistry::Global().RegisterFn("reduce_sum_test_1_kernel_ptr_",
                                                       reinterpret_cast<void*>(&reduce_sum_test_1_kernel));
  backends::RuntimeSymbolRegistry::Global().RegisterVar("reduce_sum_test_1_kernel_stream_ptr_", stream);

  auto jit = backends::SimpleJIT::Create();
  jit->Link<backends::CodeGenCUDA_Host>(module_ptx.first);

  auto fn_reduce_sum_test_1 = jit->Lookup("reduce_sum_test_1");
  CHECK(fn_reduce_sum_test_1);

  auto func_0 = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_reduce_sum_test_1);

  srand(time(NULL));
  auto buffer_x = common::BufferBuilder(Float(32), {n, c, h, w}).set_random().Build();
  auto buffer_y = common::BufferBuilder(Float(32), {h, w}).set_random().Build();

  void *dev_x = nullptr, *dev_y = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_y, buffer_y->memory_size));

  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));

  cinn_buffer_t _x;
  cinn_buffer_t _y;

  _x.memory = static_cast<uint8_t*>(dev_x);
  _y.memory = static_cast<uint8_t*>(dev_y);

  _x.memory_size = buffer_x->memory_size;
  _y.memory_size = buffer_y->memory_size;

  cinn_pod_value_t x_arg(&_x), y_arg(&_y);
  cinn_pod_value_t args0[] = {x_arg, y_arg};

  func_0(args0, 2);
  CUDA_CALL(cudaMemcpy(buffer_y->memory, dev_y, buffer_y->memory_size, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_y));
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
