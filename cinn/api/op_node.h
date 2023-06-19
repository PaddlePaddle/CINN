// Copyright (c) 2023 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;
using Attribute = cinn::utils::Attribute;

class TensorNode;

class OpNode {
 public:
  OpNode(const hlir::pass::FusionHelperBase* helper, const hlir::framework::Node* node) : helper_(helper), node_(node) {}

  OpPatternKind kind () {
    return helper_->GetOpKind(node_);
  }

  size_t InputsSize() const {
    return node_->inlinks.size();
  }

  size_t OutputsSize() const {
    return node_->outlinks.size();
  }

  TensorNode GetInput(size_t i) const;

  TensorNode GetOutput(size_t i) const;

  template <typename T>
  const T& GetAttr(const std::string& attr_name) const {
    return absl::get<T>(GetAttr(attr_name));
  }

 private:
  const Attribute& GetAttr(const std::string& attr_name) {
    return node_->attrs.attr_store.at(attr_name);
  }

  const hlir::pass::FusionHelperBase* helper_;
  const hlir::framework::Node* node_;
};

}  // namespace api
}  // namespace cinn
