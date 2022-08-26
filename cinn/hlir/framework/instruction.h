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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/hlir/framework/scope.h"
#ifdef CINN_WITH_CUDA
#include "cinn/runtime/cuda/cuda_util.h"
#endif
#include "cinn/utils/string.h"
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
  void SetLoweredFunc(void* fn_ptr, const std::string& name = "") {
    fn_ptrs_.push_back(fn_ptr);
    fn_names_.push_back(name);
  }

  // explicitly finalize the instruction, and can't append function again after call it
  void Finalize();

  void UpdateArgsCache(const std::map<std::string, cinn_pod_value_t>* name2podargs);
  /**
   * Run the Instruction.
   */
  void Run(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr,
           bool dryrun                                                 = false,
           void* stream                                                = nullptr,
           bool use_cache                                              = true);

  void PreRun(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr) {}

  int size() { return fn_ptrs_.size(); }

  std::vector<std::vector<std::string>> GetInArgs() { return in_args_; }
  std::vector<std::vector<std::string>> GetOutArgs() { return out_args_; }
  std::vector<std::string> GetFnNames() { return fn_names_; }
  void AddInArgs(const std::vector<std::string>& in_args) { in_args_.push_back(in_args); }
  void AddOutArgs(const std::vector<std::string>& out_args) { out_args_.push_back(out_args); }
  std::vector<int> attrs;
  std::vector<std::string> str_attrs;
  bool pre_run = false;
  Target target_;

 protected:
  void CheckResults(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr, void* stream = nullptr);

 private:
  bool finalized_flag_ = false;
  Scope* scope_{};
  std::string function_name_;
  std::vector<std::vector<std::string>> in_args_;
  std::vector<std::vector<std::string>> out_args_;
  std::vector<std::vector<cinn_pod_value_t>> args_cached_;

  std::vector<void*> fn_ptrs_{};
  std::vector<std::string> fn_names_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
