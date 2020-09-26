#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/object.h"
#include "cinn/common/type.h"
#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

class PaddleModelToProgram {
 public:
  explicit PaddleModelToProgram(hlir::framework::Scope* scope) : scope_(scope), program_(new Program) {
    CHECK(scope_);

    AddOpMapper_feed();
    AddOpMapper_fetch();
    AddOpMapper_mul();
    AddOpMapper_scale();
    AddOpMapper_relu();
    AddOpMapper_elementwise_add();
    AddOpMapper_conv2d();
    AddOpMapper_batchnorm();
    AddOpMapper_pool2d();
    AddOpMapper_softmax();
    AddOpMapper_relu6();
    AddOpMapper_depthwise_conv2d();
  }

  std::unique_ptr<Program> operator()(const std::string& model_dir, bool is_combined);

  // Add an Instruction to a program given a Paddle-format \p op_desc.
  void AddOp(const paddle::cpp::OpDesc& op_desc);

  // @{
  void AddOpMapper_feed();
  void AddOpMapper_fetch();
  void AddOpMapper_scale();
  void AddOpMapper_mul();
  void AddOpMapper_relu();
  void AddOpMapper_elementwise_add();
  void AddOpMapper_conv2d();
  void AddOpMapper_batchnorm();
  void AddOpMapper_pool2d();
  void AddOpMapper_softmax();
  void AddOpMapper_relu6();
  void AddOpMapper_depthwise_conv2d();
  // @}

  const std::unordered_map<std::string, Variable>& var_map() const { return var_map_; }
  const std::unordered_map<std::string, std::string>& var_model_to_program_map() { return var_model_to_program_map_; }

 protected:
  void AddVar(const std::string& name, const Variable& var);

  Variable GetVar(const std::string& name);

 private:
  std::unordered_map<std::string, std::function<void(const paddle::cpp::OpDesc&)>> op_mappers_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string> var_model_to_program_map_;
  hlir::framework::Scope* scope_{};
};

}  // namespace frontend
}  // namespace cinn
