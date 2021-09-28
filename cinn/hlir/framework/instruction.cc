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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
