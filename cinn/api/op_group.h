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

#include "cinn/api/op_node.h"

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

class OpGroup {
 public:
  OpNode(const hlir::pass::FusionHelperBase* helper, const hlir::framework::Graph::Group* group) : helper_(helper), group_(group) {}

  size_t OpSize() const {
    return group->CollectNodes().size();
  }

  OpNode GetOp(size_t index) const {
    return group->CollectNodes()[index];
  }

  size_t ProducerSize() const {
    return group->producer_groups().size();
  }
  OpGroup GetProducer(size_t index) const {
    std::vector<hlir::framework::Graph::Group*> producer_groups;
    producer_groups.reserve(ProducerSize());
    for(const auto& producer : group->producer_groups()) {
      producer_groups.push_back(producer.first.get());
    }
    return OpGroup(helper_, producer_groups[index]);
  }

  size_t ConsumerSize() const {
    return group->consumer_groups().size();
  }

  OpGroup GetConsumer(size_t index) const {
    std::vector<hlir::framework::Graph::Group*> consumer_groups;
    consumer_groups.reserve(ConsumerSize());
    for(const auto& consumer : group->consumer_groups()) {
      consumer_groups.push_back(consumer.first.get());
    }
    return OpGroup(helper_, consumer_groups[index]);
  }

 private:
  const hlir::pass::FusionHelperBase* helper_;
  const hlir::framework::Graph::Group* group_;
};

}  // namespace api
}  // namespace cinn
