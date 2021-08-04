#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cinnrt/paddle/cpp/program_desc.h"
#include "cinnrt/cinn/string.h"
#include "cinnrt/common/common.h"
#include "cinnrt/common/context.h"
#include "cinnrt/common/object.h"
#include "cinnrt/common/type.h"
#include "cinnrt/paddle/node.h"
#include "cinnrt/paddle/scope.h"
#include "cinnrt/paddle/syntax.h"

namespace cinnrt {
namespace paddle {

using cinnrt::common::Target;

class PaddleModelToProgram {
 public:
  explicit PaddleModelToProgram(cinnrt::paddle::Scope* scope, const Target& target)
      : scope_(scope), target_(target), program_(new Program) {
    CHECK(scope_);

    AddOpMapper_feed();
    AddOpMapper_fetch();
    AddOpMapper_mul();
    AddOpMapper_scale();
    AddOpMapper_relu();
    AddOpMapper_elementwise_add();
    AddOpMapper_elementwise_mul();
    AddOpMapper_conv2d();
    AddOpMapper_batchnorm();
    AddOpMapper_pool2d();
    AddOpMapper_softmax();
    AddOpMapper_relu6();
    AddOpMapper_depthwise_conv2d();
    AddOpMapper_sigmoid();
    AddOpMapper_slice();
    AddOpMapper_dropout_infer();
  }

  // Add an Instruction to a program given a Paddle-format \p op_desc.
  void AddOp(const ::cinnrt::paddle::cpp::OpDesc& op_desc);

  // @{
  void AddOpMapper_feed();
  void AddOpMapper_fetch();
  void AddOpMapper_scale();
  void AddOpMapper_mul();
  void AddOpMapper_relu();
  void AddOpMapper_elementwise_add();
  void AddOpMapper_elementwise_mul();
  void AddOpMapper_conv2d();
  void AddOpMapper_batchnorm();
  void AddOpMapper_pool2d();
  void AddOpMapper_softmax();
  void AddOpMapper_relu6();
  void AddOpMapper_depthwise_conv2d();
  void AddOpMapper_sigmoid();
  void AddOpMapper_slice();
  void AddOpMapper_dropout_infer();
  // @}

  const std::unordered_map<std::string, Variable>& var_map() const { return var_map_; }
  const std::unordered_map<std::string, std::string>& var_model_to_program_map() { return var_model_to_program_map_; }

 protected:
  void AddVar(const std::string& name, const Variable& var, bool replace = false);

  Variable GetVar(const std::string& name);

  void TransposeVar(const std::string& name);

  void ReverseHWVar(const std::string& name);

 private:
  std::unordered_map<std::string, std::function<void(const ::cinnrt::paddle::cpp::OpDesc&)>> op_mappers_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, Variable> var_map_;
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string> var_model_to_program_map_;
  cinnrt::paddle::Scope* scope_{};
  Target target_;
};

}  // namespace paddle
}  // namespace cinnrt
