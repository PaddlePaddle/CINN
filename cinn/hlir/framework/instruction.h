#pragma once

#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/scope.h"
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
              const std::vector<std::string>& out_args)
      : target_(target), scope_(scope), in_args_(in_args), out_args_(out_args) {}

  /**
   * Set compiled function address.
   * @param fn The JIT compiled function address.
   */
  void SetLoweredFunc(lower_func_ptr_t fn) { fn_ = fn; }

  void RunTest(int repeat_) {
    CHECK(fn_) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
    auto& pod_args = PreparePodArgs();
    fn_(pod_args.data(), pod_args.size());
  }

  /**
   * Run the Instruction.
   */
  void Run() {
    CHECK(fn_) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
    auto& pod_args = PreparePodArgs();
    fn_(pod_args.data(), pod_args.size());
  }
  std::vector<std::string> GetInArgs() { return in_args_; }
  std::vector<std::string> GetOutArgs() { return out_args_; }

 protected:
  std::vector<cinn_pod_value_t>& PreparePodArgs();

 private:
  Scope* scope_{};
  std::vector<std::string> in_args_;
  std::vector<std::string> out_args_;

  std::vector<cinn_pod_value_t> args_cached_;

  Target target_;

  lower_func_ptr_t fn_{};
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
