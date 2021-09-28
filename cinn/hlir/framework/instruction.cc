#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/instruction.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<cinn_pod_value_t>& Instruction::PreparePodArgs(int i) {
  if (args_cached_.size() > i) return args_cached_[i];
  common::ArgsBuilder builder;
  std::vector<std::string> all_args(in_args_[i].begin(), in_args_[i].end());
  all_args.insert(std::end(all_args), out_args_[i].begin(), out_args_[i].end());

  for (auto& arg : all_args) {
    auto* var = scope_->FindVar(arg);
    CHECK(var) << "Argument [" << arg << "] not found in the scope";

    // TODO(Superjomn) Support other types.
    auto& tensor = absl::get<Tensor>(*var);
    builder.Add(tensor->buffer());
  }

  args_cached_.emplace_back(builder.Build());
  CHECK(args_cached_.size() > i);
  return args_cached_[i];
}

Instruction::Instruction(const Target& target,
                         Scope* scope,
                         const std::vector<std::string>& in_args,
                         const std::vector<std::string>& out_args,
                         const std::string& function_name)
    : target(target), scope_(scope), in_args_({in_args}), out_args_({out_args}), function_name_(function_name) {}

void Instruction::Run() {
  if (fn_.size() > 1 && fn_.size() != in_args_.size()) {
    out_args_.back()[0] = out_args_.front()[0];
    out_args_.erase(out_args_.begin());
    in_args_.erase(in_args_.begin());
  }
#ifdef CINN_WITH_CUDNN
  auto& pod_args = PreparePodArgs(0);
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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
