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

#include "tests/benchmark/test_utils.h"

#include "cinn/backends/llvm/codegen_x86.h"
#include "cinn/common/cas.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace tests {
using ir::Tensor;
std::unique_ptr<backends::ExecutionEngine> OpBenchmarkTester::CreateExecutionEngine(const cinn::ir::Module& module) {
  auto engine = backends::ExecutionEngine::Create({});
  engine->Link<backends::CodeGenX86>(module);
  return engine;
}

void OpBenchmarkTester::TestOp(const std::string& test_name,
                               const std::vector<Tensor>& input_tensors,
                               const hlir::framework::NodeAttr& attrs,
                               const std::vector<Type>& input_types,
                               const std::vector<Type>& out_types,
                               bool use_default_stragegy) {
  auto module        = CreateCinnModule(input_tensors, attrs, out_types, use_default_stragegy);
  auto engine        = CreateExecutionEngine(module);
  auto test_func_ptr = reinterpret_cast<void (*)(void**, int32_t)>(engine->Lookup(op_name_));
  input_types_       = input_types;
  out_types_         = out_types;
  CreateBuffer();
  LOG(INFO) << "Testing " << test_name;
  cinn::utils::Timer timer;
  // ignore first execution for lazy jit component
  timer.Start();
  test_func_ptr(reinterpret_cast<void**>(all_args_.data()), all_args_.size());
  double test_op_time = timer.Stop();
  LOG(INFO) << "kernel warmup run time: " << test_op_time << " ms";
  timer.Start();
  for (int i = 0; i < repeat_; i++) {
    test_func_ptr(reinterpret_cast<void**>(all_args_.data()), all_args_.size());
  }
  test_op_time = timer.Stop() / repeat_;
  LOG(INFO) << "repeat times: " << repeat_ << ", kernel run time: " << test_op_time << " ms";
}

Module OpBenchmarkTester::CreateCinnModule(const std::vector<Tensor>& input_tensors,
                                           const hlir::framework::NodeAttr& attrs,
                                           const std::vector<Type>& out_types,
                                           bool use_default_stragegy) {
  std::vector<Tensor> outs;
  std::vector<Tensor> rets;
  poly::StageMap stages;
  std::vector<Expr> output_shape_expr;
  CHECK(!out_types.empty());
  Type type = out_types.back();
  rets      = input_tensors;

  if (use_default_stragegy) {
    auto strategy = hlir::framework::Operator::GetAttrs<hlir::framework::StrategyFunction>("CINNStrategy");
    auto op       = hlir::framework::Operator::Get(op_name_);
    CHECK(op) << op_name_ << " isn't supported yet\n";
    auto impl =
        hlir::framework::OpStrategy::SelectImpl(strategy[op](attrs, input_tensors, out_types, input_shapes_, target_));
    std::vector<common::CINNValue> temp_inputs;
    for (auto& tensor : input_tensors) {
      temp_inputs.push_back(common::CINNValue(tensor));
    }
    common::CINNValuePack C = impl->fcompute(common::CINNValuePack(temp_inputs));
    stages                  = C.back();
    C                       = impl->fschedule(C);
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      stages->InsertLazily(temp.as_tensor_ref());
      std::vector<Expr> output_shape_expr = temp.as_tensor_ref()->domain_without_reduce_axis();
      std::vector<int> output_shape;
      for (auto& shape : output_shape_expr) {
        LOG(INFO) << shape;
        output_shape.push_back(common::AutoSimplify(shape).as_int32());
      }
      output_shapes_.push_back(output_shape);
      rets.push_back(temp.as_tensor_ref());
    }
  } else {
    stages = CreateStages(input_tensors);
    outs   = CreateSpecificStrategy(input_tensors, &stages);

    for (auto& out : outs) {
      stages->InsertLazily(out);
      rets.push_back(out);
      std::vector<Expr> output_shape_expr = out->domain_without_reduce_axis();
      std::vector<int> output_shape;
      for (auto& shape : output_shape_expr) {
        output_shape.push_back(shape.as_int32());
      }
      output_shapes_.push_back(output_shape);
    }
  }
  auto func = Lower(op_name_, stages, rets);
  LOG(INFO) << "After Lower, func is: \n" << func;
  Module::Builder builder("module_" + op_name_, target_);
  builder.AddFunction(func);
  CodeGenC compiler(target_);
  Outputs outputs;
  outputs = outputs.c_header("./test_" + op_name_ + ".h").c_source("./test_" + op_name_ + ".cc");
  compiler.Compile(builder.Build(), outputs);
  return builder.Build();
}

void OpBenchmarkTester::CreateBuffer() {
  std::vector<cinn_pod_value_t> args;
  for (size_t i = 0; i < input_shapes_.size(); i++) {
    auto* buffer = common::BufferBuilder(input_types_[i], input_shapes_[i]).set_align(32).set_random().Build();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
  }
  CHECK(!output_shapes_.empty()) << "output shapes shouldn't be empty\n";
  CHECK_EQ(output_shapes_.size(), out_types_.size());
  for (size_t i = 0; i < output_shapes_.size(); i++) {
    if (out_types_[i].is_void()) continue;
    auto* buffer = common::BufferBuilder(out_types_[i], output_shapes_[i]).set_align(32).set_zero().Build();
    CHECK(buffer);
    out_dims_ = buffer->num_elements();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
  }
}

}  // namespace tests
}  // namespace cinn
