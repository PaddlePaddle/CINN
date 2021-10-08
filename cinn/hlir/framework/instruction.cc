#include "cinn/hlir/framework/instruction.h"
#include "cinn/common/test_helper.h"

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

void Instruction::AlignArgs() {
  if (fn_.size() > 1 && fn_.size() != in_args_.size()) {
    out_args_.back()[0] = out_args_.front()[0];
    out_args_.erase(out_args_.begin());
    in_args_.erase(in_args_.begin());
  }
}

void Instruction::RunInOrder(std::vector<std::vector<cinn_pod_value_t>>* arguments_array) {
  CHECK_EQ(pod_arguments->size(), fn_.size()) << "The number of Argument array should be euqal to LoweredFunc array";
#ifdef CINN_WITH_CUDNN
  auto& pod_args = arguments_array->front();
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
      auto& pod_args = arguments_array->at(i);
      CHECK(it_fn) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
      it_fn(pod_args.data(), pod_args.size());
      i++;
    }
  }
#else
  int i = 0;
  for (auto& it_fn : fn_) {
    auto& pod_args = arguments_array->at(i);
    CHECK(it_fn) << "The LoweredFunc address should be set first by calling SetLoweredFunc method";
    it_fn(pod_args.data(), pod_args.size());
    i++;
  }
#endif
}

void Instruction::Run() {
  AlignArgs();

  for (auto i = 0; i < fn_.size(); ++i) {
    PreparePodArgs(i);
  }
  RunInOrder(&args_cached_);
}

void Instruction::Run(const std::unordered_map<std::string, cinn_pod_value_t>& name2podargs) {
  AlignArgs();

  std::vector<std::vector<cinn_pod_value_t>> arguments_array;
  for (auto i = 0; i < fn_.size(); ++i) {
    std::vector<cinn_pod_value_t> arguments;
    std::vector<std::string> all_args(in_args_[i].begin(), in_args_[i].end());
    all_args.insert(std::end(all_args), out_args_[i].begin(), out_args_[i].end());
    for (auto& arg : all_args) {
      auto it = name2podargs.find(arg);
      CHECK_NE(it, name2podargs.end()) << "Argument [" << arg << "] not found in the name2podargs";
      arguments.emplace_back(*it);
    }
    arguments_array.emplace_back(std::move(arguments);
  }
  RunInOrder(&arguments_array);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
