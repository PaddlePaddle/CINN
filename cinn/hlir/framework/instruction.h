#pragma once

#include <map>
#include <string>
#include <utility>
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
  void Run(const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr);

  std::vector<std::vector<std::string>> GetInArgs() { return in_args_; }
  std::vector<std::vector<std::string>> GetOutArgs() { return out_args_; }
  void AddInArgs(const std::vector<std::string>& in_args) { in_args_.push_back(in_args); }
  void AddOutArgs(const std::vector<std::string>& out_args) { out_args_.push_back(out_args); }
  std::vector<int> attrs;
  std::vector<std::string> str_attrs;
  bool pre_run = false;
  Target target_;

 protected:
  std::vector<cinn_pod_value_t>& PreparePodArgs(int i, const std::map<std::string, cinn_pod_value_t>* name2podargs);

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
