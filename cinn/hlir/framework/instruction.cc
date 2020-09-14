#include "cinn/hlir/framework/instruction.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<cinn_pod_value_t>& Instruction::PreparePodArgs() {
  if (!args_cached_.empty()) return args_cached_;

  common::ArgsBuilder builder;
  std::vector<std::string> all_args(in_args_.begin(), in_args_.end());
  all_args.insert(std::end(all_args), out_args_.begin(), out_args_.end());

  for (auto& arg : all_args) {
    auto* var = scope_->FindVar(arg);
    CHECK(var) << "Argument [" << arg << "] not found in the scope";

    // TODO(Superjomn) Support other types.
    auto& tensor = std::get<Tensor>(*var);
    builder.Add(tensor->buffer());
  }

  args_cached_ = builder.Build();
  return args_cached_;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
