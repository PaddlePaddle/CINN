// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/framework/pass.h"

namespace cinn {
namespace hlir {
namespace pass {

class ConvertStage {
 public:
  virtual bool CanConvert(const framework::Node*) = 0;
  virtual void TryConvert(const framework::Node*, framework::Graph*) = 0;
};

class ConvertPipeline {
 public:
  template <typename T>
  void AddStage() {
    stages_.push_back(std::make_unique<T>());
  }

  void Run(framework::Graph* graph) {
    auto grapd_nodes = std::get<0>(graph->topological_order());
    for (auto graph_node : grapd_nodes) {
      auto op_node = graph_node->safe_as<framework::Node>();
      if (!op_node) continue;
      for (auto& stage : stages_) {
        if (stage->CanConvert(op_node)) {
          stage->TryConvert(op_node, graph);
        }
      }
    }
  }

 private:
  std::vector<std::unique_ptr<ConvertStage>> stages_;
};

class ConvertAddRelu : public ConvertStage {
 public:
  bool CanConvert(const framework::Node* op_node) override {
    return "elementwise_add" == op_node->attrs.node_name;
  }

  void TryConvert(const framework::Node* ewadd, framework::Graph* graph) override {
    auto ewadd_out = ewadd->outlinks_in_order().front()->sink()->safe_as<framework::NodeData>();
    if (ewadd_out->outlinks().size() == 1) return;
    std::vector<framework::Node*> relu_nodes;
    for (auto link : ewadd_out->outlinks()) {
      auto link_op = link->sink()->safe_as<framework::Node>();
      if ("relu" == link_op->attrs.node_name) {
        relu_nodes.push_back(link_op);
      }
    }
    size_t start_index = 0;
    if (ewadd_out->outlinks().size() == relu_nodes.size()) {
      start_index = 1;
    }
    for (size_t i = start_index; i < relu_nodes.size(); ++i) {
      framework::Node* node = new framework::Node(ewadd->op(), ewadd->attrs.node_name, ewadd->id() + "_" + std::to_string(i));
      framework::NodeData* node_data = new framework::NodeData(std::shared_ptr<framework::Node>(node), 0, 0, ewadd_out->id() + "_" + std::to_string(i));
      for (auto ewadd_inlink : ewadd->inlinks()) {
        ewadd_inlink->source()->LinkTo(node);
      }
      node->LinkTo(node_data);
      node_data->LinkTo(relu_nodes[i]);
      graph->RegisterNode(node_data->id(), node_data);
      graph->RegisterNode(node->id(), node);
      ewadd_out->UnLinkTo(relu_nodes[i]);
    }
  }
};

void ConvertToMoreFusionPass(framework::Graph* graph) {
  ConvertPipeline pipeline;
  pipeline.AddStage<ConvertAddRelu>();
  pipeline.Run(graph);
}

} // namespace pass
} // namespace hlir
} // namespace cinn

CINN_REGISTER_HELPER(ConvertToMoreFusion) {
  CINN_REGISTER_PASS(ConvertToMoreFusion)
      .describe("This pass")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::ConvertToMoreFusionPass);
  return true;
}
