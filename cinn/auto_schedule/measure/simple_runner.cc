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

#include "cinn/auto_schedule/measure/simple_runner.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>
#include <random>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/buffer.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace auto_schedule {

using hlir::framework::Buffer;
using hlir::framework::Shape;
using hlir::framework::Tensor;

// Generate random value and populate them to the output address of memeory
static void PopulateRandomValue(const common::Type& type, const int numel, void* raw_ptr) {
  std::random_device seed;
  std::default_random_engine engine(seed());

  if (type == common::Bool()) {
    auto* fmt_ptr = reinterpret_cast<bool*>(raw_ptr);
    std::bernoulli_distribution dist(0.5);
    std::generate_n(fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else if (type == common::I32()) {
    auto* fmt_ptr = reinterpret_cast<int*>(raw_ptr);
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    std::generate_n(fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else if (type == common::F32()) {
    auto* fmt_ptr = reinterpret_cast<float*>(raw_ptr);
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    std::generate_n(fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else {
    LOG(FATAL) << "Unsupported type:" << type;
  }
}

// Alloc a new buffer in specificed target with initial infos.
static std::shared_ptr<Buffer> AllocBuffer(const common::Target& target,
                                           const common::Type& type,
                                           const Shape& shape,
                                           bool fill_random_value = true) {
  static constexpr int default_alignment = 1024;
  auto buffer                            = std::make_shared<Buffer>(target);

  VLOG(6) << "AllocBuffer target:" << target << ", type:" << type << ", numel:" << shape.numel()
          << ", fill_random_value:" << fill_random_value;
  if (target == common::DefaultHostTarget()) {
    buffer->ResizeLazy(default_alignment, shape.numel() * type.bytes());
  } else {
    buffer->ResizeLazy(shape.numel() * type.bytes());
  }

  return buffer;
}

SimpleRunner::SimpleRunner(int repeat_times) : repeat_times_(repeat_times) {
  CHECK_GT(repeat_times_, 0) << "repeat_times can't less than 0";
}

// Prepare execution arguments of all instructions to run, a argument
// may be obtained from the input of measurement or allocating new buffer
// with random value.
std::map<std::string, cinn_pod_value_t> SimpleRunner::PrepareArgs(const MeasureInput& input,
                                                                  const BuildResult& build_result,
                                                                  hlir::framework::Scope* temp_scope) {
  std::map<std::string, cinn_pod_value_t> result;

  const auto& target         = input.task->target;
  const auto* input_args     = input.execution_args;
  const auto* compiled_scope = build_result.compiled_scope;
  const auto& instructions   = build_result.runtime_program->GetRunInstructions();

  auto fill_arg_fn = [&](const std::string& param) {
    VLOG(6) << "Filling argument:" << param;
    // the argument is duplicated and has been prepared.
    if (result.count(param)) {
      return;
    }

    // if the input of measurement specifies this argument,
    // we should use it firstly.
    if (input_args && input_args->count(param)) {
      VLOG(6) << "Argument[" << param << "] use input value";
      result.emplace(param, input_args->at(param));
      return;
    }

    if (temp_scope->FindVar(param)) {
      auto temp_tensor = temp_scope->GetTensor(param);
      result.emplace(param, temp_tensor->buffer());
      return;
    }

    // allocate a new buffer for this argument and store it in
    // the temporary scope to be released at proper time.
    auto compiled_tensor = compiled_scope->GetTensor(param);
    auto buffer          = AllocBuffer(target, compiled_tensor->type(), compiled_tensor->shape());
    temp_scope->Var<Tensor>(param);
    auto temp_tensor = temp_scope->GetTensor(param);
    temp_tensor->set_buffer(buffer);
    result.emplace(param, temp_tensor->buffer());
  };

  for (auto&& instr : instructions) {
    for (auto&& args : instr->GetInArgs()) {
      std::for_each(args.begin(), args.end(), fill_arg_fn);
    }

    for (auto&& args : instr->GetOutArgs()) {
      std::for_each(args.begin(), args.end(), fill_arg_fn);
    }
  }
  return result;
}

MeasureResult SimpleRunner::Run(const MeasureInput& input, const BuildResult& build_result) {
  MeasureResult result;
  auto t_start = std::chrono::steady_clock::now();
  // prepare execution arguments
  VLOG(4) << "SimpleRunner prepare execution arguments";
  hlir::framework::Scope temp_scope;  // used for store temporary allocated data
  auto execution_args = PrepareArgs(input, build_result, &temp_scope);

  // Execute each instruction repeatedly and take the average as cost.
  result.execution_cost    = 0;
  const auto& instructions = build_result.runtime_program->GetRunInstructions();
  for (auto ct = 0; ct < instructions.size(); ++ct) {
    auto&& instr = instructions.at(ct);
    VLOG(5) << "Start running instruction-" << ct;
    auto run_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat_times_; ++i) {
      instr->Run(&execution_args);
    }
#ifdef CINN_WITH_CUDA
    if (instr->target_ == common::DefaultNVGPUTarget()) {
      CUDA_CALL(cudaDeviceSynchronize());
    }
#endif
    auto time_span =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - run_start);
    auto cost_avg = static_cast<double>(time_span.count()) / repeat_times_;
    result.execution_cost += cost_avg;
  }

  auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start);
  result.elapsed_time = static_cast<double>(time_span.count());

  VLOG(4) << "A measurement done:repeat_times[" << repeat_times_ << "]total_elapsed_time[" << result.elapsed_time
          << "]us,execution_cost[" << result.execution_cost << "]us";
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn
