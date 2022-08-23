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

#pragma once

#include <string>
#include <unordered_map>

namespace cinn::hlir::pass::utils {

std::string GenerateCheckFusionAccuracyNodeId(const std::string& node_id);

bool IsCheckFusionAccuracyNode(const std::string& node_id);

bool IsCheckFusionAccuracyNodeGenerated(const std::string& check_node_id, const std::string& node_id);

class AssertMsg {
 public:
  AssertMsg(const std::string& introduction) : introduction_(introduction) {}

  void SetMsg(const std::string& title, const std::string& msg) { msg_info_[title] = msg; }

  void CleasMsg(const std::string& title) { msg_info_.erase(title); }

  std::string str() const;

 private:
  std::string introduction_;
  std::unordered_map<std::string, std::string> msg_info_;
};

}  // namespace cinn::hlir::pass::utils
