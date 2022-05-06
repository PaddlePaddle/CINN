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

std::pair<ir::Module, std::string> GenReduceCode(const std::vector<int>& shape,
                                                 const std::vector<int>& dim,
                                                 const std::string& func_name,
                                                 bool keep_dim              = false,
                                                 const std::string& op_name = "reduce_sum") {
  // code gen
  auto reduce_sum = Operator::Get(op_name);
  auto strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy")[reduce_sum];

  // input tensor
  std::vector<Expr> shape_as_expr;
  for (auto value : shape) {
    shape_as_expr.emplace_back(value);
  }
  Placeholder<float> X("X", shape_as_expr);

  // set attrs
  NodeAttr attrs;
  attrs.attr_store["dim"]      = dim;
  attrs.attr_store["keep_dim"] = keep_dim;
  std::vector<ir::Tensor> inputs{X.tensor()};
  std::vector<Type> out_type{Float(32)};

  std::vector<int> output_shape;
  for (int idx = 0; idx < shape.size(); ++idx) {
    if (std::find(dim.begin(), dim.end(), idx) != dim.end()) {
      if (keep_dim) {
        output_shape.push_back(1);
      }
    } else {
      output_shape.push_back(shape[idx]);
    }
  }

  auto target = common::DefaultNVGPUTarget();
  auto impl   = OpStrategy::SelectImpl(strategy(attrs, inputs, out_type, {output_shape}, target));

  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(X)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  poly::StageMap stages            = rets.back();

  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    if (!temp.as_tensor_ref()->buffer.defined() && !stages[temp.as_tensor_ref()]->inlined()) {
      inputs.push_back(temp.as_tensor_ref());
    }
  }

  auto func = lang::LowerVec(func_name, rets.back(), inputs, {}, {}, nullptr, target);
  for (auto& f : func) {
    LOG(INFO) << "Test Strategy Codegen:\n" << f;
  }

  Module::Builder builder(func_name + "_builder", target);
  for (auto& f : func) {
    builder.AddFunction(f);
  }

  auto module                    = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module              = std::get<0>(host_module_device_module);
  auto& device_module            = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n" << source_code;

  return std::pair<ir::Module, std::string>(host_module, source_code);
}

TEST(Operator, Operator_Reduction_Case_0) {
  std::vector<int> shape = {16, 16, 8, 16};
  std::vector<int> dim   = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_0");
}

TEST(Operator, Operator_Reduction_Case_0_0) {
  std::vector<int> shape = {16, 16, 8, 16};
  std::vector<int> dim   = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_0_0", true);
}

TEST(Operator, Operator_Reduction_Case_1) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim   = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_1");
}

TEST(Operator, Operator_Reduction_Case_1_1) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim   = {2, 3};

  GenReduceCode(shape, dim, "reduce_cast_1_1", true);
}

TEST(Operator, Operator_Reduction_Case_2) {
  std::vector<int> shape = {16, 16, 32, 32};
  std::vector<int> dim   = {1};

  GenReduceCode(shape, dim, "reduce_cast_2", true);
}

TEST(Operator, Operator_Reduction_Case_3) {
  std::vector<int> shape = {16, 16, 64, 64};
  std::vector<int> dim   = {1};

  GenReduceCode(shape, dim, "reduce_cast_3");
}

TEST(Operator, Operator_Reduction_Case_4) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim   = {0, 2, 3};

  GenReduceCode(shape, dim, "reduce_cast_4");
}

TEST(Operator, Operator_Reduction_Case_4_4) {
  std::vector<int> shape = {16, 16, 16, 16};
  std::vector<int> dim   = {0, 2, 3};

  GenReduceCode(shape, dim, "reduce_cast_4_4", true);
}

TEST(Operator, Operator_Reduction_Case_5) {
  std::vector<int> shape = {16, 16, 16, 16, 16, 32};
  std::vector<int> dim   = {1, 3, 5};

  GenReduceCode(shape, dim, "reduce_cast_5");
}

TEST(Operator, Operator_Reduction_Case_5_5) {
  std::vector<int> shape = {16, 16, 16, 16, 16, 32};
  std::vector<int> dim   = {1, 3, 5};

  GenReduceCode(shape, dim, "reduce_cast_5_5", true);
}

TEST(Operator, Operator_Reduction_Case_6_0) {
  std::vector<int> shape = {32, 32, 32};
  std::vector<int> dim   = {0, 1, 2};

  GenReduceCode(shape, dim, "reduce_cast_6_0", false);
}

TEST(Operator, Operator_Reduction_Case_6_00) {
  std::vector<int> shape = {32, 32, 32, 32};
  std::vector<int> dim   = {0, 1, 2};

  GenReduceCode(shape, dim, "reduce_cast_6_00", false);
}

struct SumOp {
  float operator()(const float left, const float right) { return left + right; }
};
struct ProdOp {
  float operator()(const float left, const float right) { return left * right; }
};
struct MaxOp {
  float operator()(const float left, const float right) { return std::max(left, right); }
};
struct MinOp {
  float operator()(const float left, const float right) { return std::min(left, right); }
};

template <class Op>
void DoCpuReduce(const float* x,
                 std::vector<float>* sum0,
                 std::vector<float>* sum1,
                 const float init_val,
                 const int n,
                 const int c,
                 const int h,
                 const int w) {
  for (auto& val : *sum0) {
    val = init_val;
  }
  for (auto& val : *sum1) {
    val = init_val;
  }

  for (int idx = 0; idx < n; ++idx) {
    for (int idy = 0; idy < c; ++idy) {
      for (int idz = 0; idz < h; ++idz) {
        for (int ida = 0; ida < w; ++ida) {
          sum0->at(idy * w + ida) += Op()(sum0->at(idy * w + ida), x[idx * c * h * w + idy * h * w + idz * w + ida]);
          sum1->at(idy) = Op()(sum1->at(idy), x[idx * c * h * w + idy * h * w + idz * w + ida]);
        }
      }
    }
  }
}

template <class Op>
void TestCaseForReduce(
    const float init_val, int n, int c, int h, int w, const std::string& test_name, const std::string& op_name) {
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim   = {0, 2, 3};

  // get source code
  auto source_code = GenReduceCode(shape, dim, test_name, false, op_name).second;

  // nv jit compile to ptx
  backends::NVRTC_Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());

  // cuda_module load ptx
  runtime::cuda::CUDAModule cuda_module(ptx, CUDAModule::Kind::PTX);

  srand(time(NULL));
  CUDA_CALL(cudaSetDevice(0));

  // auto func_0   = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_reduce_sum);
  auto buffer_x = common::BufferBuilder(Float(32), {n, c, h, w}).set_random().Build();
  auto buffer_z = common::BufferBuilder(Float(32), {c}).set_random().Build();

  void *dev_x = nullptr, *dev_z = nullptr;
  CUDA_CALL(cudaMalloc(&dev_x, buffer_x->memory_size));
  CUDA_CALL(cudaMalloc(&dev_z, buffer_z->memory_size));
  CUDA_CALL(cudaMemcpy(dev_x, buffer_x->memory, buffer_x->memory_size, cudaMemcpyHostToDevice));

  dim3 grid(n * c, 1, 1);
  dim3 block(h * w, 1, 1);
  void* args[] = {&dev_x, &dev_z};

  cuda_module.LaunchKernel(0, test_name, grid, block, args);
  CUDA_CALL(cudaMemcpy(buffer_z->memory, dev_z, buffer_z->memory_size, cudaMemcpyDeviceToHost));

  std::vector<float> sum0(c * w);
  std::vector<float> sum1(c);
  DoCpuReduce<Op>(reinterpret_cast<float*>(buffer_x->memory), &sum0, &sum1, init_val, n, c, h, w);

  std::vector<std::pair<std::vector<float>, float*>> results = {{sum1, reinterpret_cast<float*>(buffer_z->memory)}};
  for (auto& res : results) {
    for (int idx = 0; idx < res.first.size(); ++idx) {
      ASSERT_LT(abs(res.first[idx] - res.second[idx]) / res.first[idx], 1e-4);
    }
  }

  CUDA_CALL(cudaFree(dev_x));
  CUDA_CALL(cudaFree(dev_z));
}

TEST(Operator, Operator_Reduction_Case_6_1) {
  TestCaseForReduce<SumOp>(0.0f, 32, 32, 32, 32, "Operator_Reduction_Case_6_1", "reduce_sum");
}
TEST(Operator, Operator_Reduction_Case_6_2) {
  TestCaseForReduce<ProdOp>(1.0f, 1, 1, 1, 32, "Operator_Reduction_Case_6_2", "reduce_prod");
}
TEST(Operator, Operator_Reduction_Case_6_3) {
  TestCaseForReduce<MaxOp>(-1e38f, 32, 32, 32, 32, "Operator_Reduction_Case_6_3", "reduce_max");
}
TEST(Operator, Operator_Reduction_Case_6_4) {
  TestCaseForReduce<MinOp>(1e38f, 32, 32, 32, 32, "Operator_Reduction_Case_6_4", "reduce_min");
}

TEST(Operator, Operator_Reduction_Case_7) {
  int n = 32, c = 32, h = 16, w = 16;
  std::vector<int> shape = {n, c, h, w};
  std::vector<int> dim   = {0, 1};

  std::string func_name = "reduce_cast_7";
  // get source code
  auto host_source = GenReduceCode(shape, dim, func_name);

  // compile to ptx
  backends::NVRTC_Compiler compiler;
  auto ptx = compiler(host_source.second);
  CHECK(!ptx.empty());

  // load ptx
  CUDA_CALL(cudaSetDevice(0));
  runtime::cuda::CUDAModule cuda_module(ptx, runtime::cuda::CUDAModule::Kind::PTX);
  void* reduce_sum_kernel = cuda_module.GetFunction(0, func_name);
  CHECK(reduce_sum_kernel);

  // register cufunction and stream
  void* stream = nullptr;
  backends::RuntimeSymbolRegistry::Global().RegisterFn(func_name + "_kernel_ptr_",
                                                       reinterpret_cast<void*>(&reduce_sum_kernel));
  backends::RuntimeSymbolRegistry::Global().RegisterVar(func_name + "_kernel_stream_ptr_", stream);

  // gen host code
  auto jit = backends::SimpleJIT::Create();
  jit->Link<backends::CodeGenCUDA_Host>(host_source.first);

  auto fn_reduce_sum = jit->Lookup(func_name);
  CHECK(fn_reduce_sum);

  auto func_0 = reinterpret_cast<void (*)(cinn_pod_value_t*, int)>(fn_reduce_sum);

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

TEST(Operator, Operator_Reduction_Case_8) {
  std::vector<int> shape = {128, 1};
  std::vector<int> dim   = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_8");
}

TEST(Operator, Operator_Reduction_Case_88) {
  std::vector<int> shape = {128, 1};
  std::vector<int> dim   = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_88", true);
}

TEST(Operator, Operator_Reduction_Case_9) {
  std::vector<int> shape = {2560, 1};
  std::vector<int> dim   = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_9");
}

TEST(Operator, Operator_Reduction_Case_99) {
  std::vector<int> shape = {2560, 1};
  std::vector<int> dim   = {0};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_99", true);
}

TEST(Operator, Operator_Reduction_Case_10) {
  std::vector<int> shape = {16, 2560, 1};
  std::vector<int> dim   = {1};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_10");
}

TEST(Operator, Operator_Reduction_Case_11) {
  std::vector<int> shape = {16, 128, 128, 1};
  std::vector<int> dim   = {1, 2};

  GenReduceCode(shape, dim, "Operator_Reduction_Case_11");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
