#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

class PaddleModelToNetBuilder {
 public:
  explicit PaddleModelToNetBuilder(hlir::framework::Scope* scope, const common::Target& target)
      : scope_(scope), target_(target) {
    CHECK(scope_);
  }

  static void RunOp(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx);

  std::unique_ptr<NetBuilder> operator()(const std::string& model_dir, bool is_combined);

  const std::unordered_map<std::string, Variable>& var_map() const { return var_map_; }
  const std::unordered_map<std::string, std::string>& var_model_to_program_map() { return var_model_to_program_map_; }

 private:
  std::unordered_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string> var_model_to_program_map_;
  hlir::framework::Scope* scope_{};
  const common::Target& target_;
};

}  // namespace frontend
}  // namespace cinn
