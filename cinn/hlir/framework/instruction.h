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

#pragma once

#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/hlir/framework/scope.h"
#ifdef CINN_WITH_CUDNN
#include "cinn/runtime/cuda/cuda_util.h"
#endif
#include "cinn/utils/timer.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * Instruction is the basic executable element in runtime, it holds a pointer to the JIT-compiled LoweredFunc, and
 * collect the cinn_buffer of the inputs and outputs from the scope, prepare the arguments and finally pass them into
 * the LoweredFunc and execute it.
 */
class Instruction {
 public:
  using infershape_t = std::function<void(Scope*, const std::vector<std::string>&)>;

  /**
   * Constructor.
   * @param target The \p target the instruction runs on.
   * @param scope The scope containing all the runtime variables(Tensors and PODs).
   * @param in_args The names of the inputs.
   * @param out_args The names of the outputs.
   * @param infershape The handler of this Instruction to perform shape inference.
   */
  Instruction(const Target& target,
              Scope* scope,
              const std::vector<std::string>& in_args,
              const std::vector<std::string>& out_args,
              const std::string& function_name = "")
      : target_(target), scope_(scope), in_args_({in_args}), out_args_({out_args}), function_name_(function_name) {}

  /**
   * Set compiled function address.
   * @param fn The JIT compiled function address.
   */
  void SetLoweredFunc(lower_func_ptr_t fn) { fn_.push_back(fn); }

  /**
   * Run the Instruction.
   */
  void Run() {
    if (fn_.size() > 1 && fn_.size() != in_args_.size()) {
      out_args_.back()[0] = out_args_.front()[0];
      out_args_.erase(out_args_.begin());
      in_args_.erase(in_args_.begin());
    }
#ifdef CINN_WITH_CUDNN
    auto& pod_args = PreparePodArgs(0);
    // Here conv2d and depthwise_conv2d are implemented by one cudnn api cudnnConvolutionForward
    if ((function_name_ == "conv2d" || function_name_ == "depthwise_conv2d") && target_.arch == Target::Arch::NVGPU) {
      absl::flat_hash_map<std::string, int> attrs_map = {
          {"input_n", attrs[0]},     {"input_c", attrs[1]},     {"input_h", attrs[2]},   {"input_w", attrs[3]},
          {"weights_n", attrs[4]},   {"weights_c", attrs[5]},   {"weights_h", attrs[6]}, {"weights_w", attrs[7]},
          {"pad_h", attrs[8]},       {"pad_w", attrs[9]},       {"stride_h", attrs[10]}, {"stride_w", attrs[11]},
          {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]}, {"groups", attrs[14]},   {"output_n", attrs[15]},
          {"output_c", attrs[16]},   {"output_h", attrs[17]},   {"output_w", attrs[18]},
      };
      if (str_attrs[0] == "forward") {
        // input weight output
        runtime::cuda::cinn_gpu_cudnn_conv2d(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
      } else if (str_attrs[0] == "backward_data") {
        // weight dy dx
        runtime::cuda::cinn_gpu_cudnn_conv2d_backward_data(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
      } else {
        // input dy dx
        runtime::cuda::cinn_gpu_cudnn_conv2d_backward_filter(attrs_map, pod_args[0], pod_args[1], pod_args[2]);
      }
    } else if (function_name_ == "pool2d" && target_.arch == Target::Arch::NVGPU) {
      runtime::cuda::cinn_gpu_cudnn_pool2d(attrs, str_attrs, pod_args[0], pod_args[1]);
    } else if (function_name_ == "softmax" && target_.arch == Target::Arch::NVGPU) {
      CHECK_EQ(pod_args.size(), 3);
      runtime::cuda::cinn_gpu_cudnn_softmax(attrs, pod_args[0], pod_args[1]);
    } else if (function_name_ == "mul" && target_.arch == Target::Arch::NVGPU) {
      CHECK_EQ(pod_args.size(), 4);
      runtime::cuda::cinn_gpu_cublas_mul(attrs, pod_args[0], pod_args[1], pod_args[2]);
    } else {
      int i = 0;
      for (auto& it_fn : fn_) {
        auto& pod_args = PreparePodArgs(i);
        CHECK(it_fn) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
        it_fn(pod_args.data(), pod_args.size());
        i++;
      }
    }
#else
    int i = 0;
    for (auto& it_fn : fn_) {
      auto& pod_args = PreparePodArgs(i);
      CHECK(it_fn) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
      it_fn(pod_args.data(), pod_args.size());
      i++;
    }
#endif
  }
  std::vector<std::vector<std::string>> GetInArgs() { return in_args_; }
  std::vector<std::vector<std::string>> GetOutArgs() { return out_args_; }
  void AddInArgs(const std::vector<std::string>& in_args) { in_args_.push_back(in_args); }
  void AddOutArgs(const std::vector<std::string>& out_args) { out_args_.push_back(out_args); }
  std::vector<int> attrs;
  std::vector<std::string> str_attrs;
  bool pre_run = false;
  Target target_;

 protected:
  std::vector<cinn_pod_value_t>& PreparePodArgs(int i);

 private:
  Scope* scope_{};
  std::string function_name_;
  std::vector<std::vector<std::string>> in_args_;
  std::vector<std::vector<std::string>> out_args_;

  std::vector<std::vector<cinn_pod_value_t>> args_cached_;

  std::vector<lower_func_ptr_t> fn_{};
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
