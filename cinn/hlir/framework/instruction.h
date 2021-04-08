#pragma once

#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/test_helper.h"
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
      : target_(target), scope_(scope), in_args_(in_args), out_args_(out_args), function_name_(function_name) {}

  /**
   * Set compiled function address.
   * @param fn The JIT compiled function address.
   */
  void SetLoweredFunc(lower_func_ptr_t fn) { fn_ = fn; }

  /**
   * Run the Instruction.
   */
  void Run() {
    CHECK(fn_) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
    auto& pod_args = PreparePodArgs();
#ifdef CINN_WITH_CUDNN
    // Here conv2d and depthwise_conv2d are implemented by one cudnn api cudnnConvolutionForward
    if ((function_name_ == "conv2d" || function_name_ == "depthwise_conv2d") && target_.arch == Target::Arch::NVGPU) {
      runtime::cuda::cinn_gpu_cudnn_conv2d(attrs, pod_args[0], pod_args[1], pod_args[2]);
    } else if (function_name_ == "pool2d" && target_.arch == Target::Arch::NVGPU) {
      runtime::cuda::cinn_gpu_cudnn_pool2d(attrs, str_attrs, pod_args[0], pod_args[1]);
    } else if (function_name_ == "softmax" && target_.arch == Target::Arch::NVGPU) {
      CHECK_EQ(pod_args.size(), 3);
      runtime::cuda::cinn_gpu_cudnn_softmax(attrs, pod_args[0], pod_args[1]);
    } else if (function_name_ == "mul" && target_.arch == Target::Arch::NVGPU) {
      CHECK_EQ(pod_args.size(), 4);
      runtime::cuda::cinn_gpu_cublas_mul(attrs, pod_args[0], pod_args[1], pod_args[2]);
    } else {
      fn_(pod_args.data(), pod_args.size());
    }
#else
    fn_(pod_args.data(), pod_args.size());
#endif
  }
  std::vector<std::string> GetInArgs() { return in_args_; }
  std::vector<std::string> GetOutArgs() { return out_args_; }
  std::vector<int> attrs;
  std::vector<std::string> str_attrs;
  Target target_;

 protected:
  std::vector<cinn_pod_value_t>& PreparePodArgs();

 private:
  Scope* scope_{};
  std::string function_name_;
  std::vector<std::string> in_args_;
  std::vector<std::string> out_args_;

  std::vector<cinn_pod_value_t> args_cached_;

  lower_func_ptr_t fn_{};
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
