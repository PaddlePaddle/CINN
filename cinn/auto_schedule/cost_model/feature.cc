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

#include "cinn/auto_schedule/cost_model/feature.h"

#include <glog/logging.h>

#include <vector>

namespace cinn {
namespace auto_schedule {

Feature::Feature()
    : stack_encoded_feature_(1),  // initialze a LoopBlockFeature as root block
      current_loop_block_index_(0),
      parent_indices_(1, -1) {}

std::vector<float> Feature::ToVector() { return std::vector<float>(); }

void Feature::IntoLoopBlock() {
  stack_encoded_feature_.emplace_back(LoopBlockFeature());
  stack_encoded_feature_[current_loop_block_index_].num_sub_loops += 1;
  parent_indices_.push_back(current_loop_block_index_);
  current_loop_block_index_ = stack_encoded_feature_.size() - 1;
}

void Feature::ExitLoopBlock() { current_loop_block_index_ = parent_indices_[current_loop_block_index_]; }

LoopBlockFeature& Feature::CurrentLoopBlock() { return stack_encoded_feature_[current_loop_block_index_]; }

const LoopBlockFeature& Feature::CurrentLoopBlock() const { return stack_encoded_feature_[current_loop_block_index_]; }

}  // namespace auto_schedule
}  // namespace cinn
