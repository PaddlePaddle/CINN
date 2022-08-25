// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/pass/check_fusion_accuracy_pass_util.h"

#include "cinn/common/context.h"

namespace cinn::hlir::pass::utils {

namespace {
constexpr char check_node_suffix[] = "_acc_check";
}

std::string GenerateCheckFusionAccuracyNodeId(const std::string& node_id) {
  return node_id + cinn::common::UniqName(check_node_suffix);
}

bool IsCheckFusionAccuracyNode(const std::string& node_id) {
  return node_id.find(check_node_suffix) != std::string::npos;
}

bool IsCheckFusionAccuracyNodeGenerated(const std::string& check_node_id, const std::string& node_id) {
  return check_node_id.find(node_id + check_node_suffix) != std::string::npos;
}

std::string AssertMsg::str() const {
  std::string format_str = introduction_ + "\n";
  for (const auto& msg_pair : msg_info_) {
    format_str += "\t\t- " + msg_pair.first + ": " + msg_pair.second + "\n";
  }
  return format_str;
}

}  // namespace cinn::hlir::pass::utils
