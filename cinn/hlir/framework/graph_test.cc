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

#include "cinn/hlir/framework/graph.h"

#include <gtest/gtest.h>

#include "cinn/common/target.h"
#include "cinn/frontend/decomposer/test_helper.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"

DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using ShapeDict = absl::flat_hash_map<std::string, framework::shape_t>;
using DtypeDict = absl::flat_hash_map<std::string, common::Type>;

TEST(Graph, visualize) {
  frontend::NetBuilder builder("test");
  auto x            = builder.CreateInput(Float(32), {32, 16}, "x");
  auto y            = builder.CreateInput(Float(32), {32, 16}, "y");
  auto add_1        = builder.ElementwiseAdd(x, y);
  auto relu_1       = builder.Relu(add_1);
  auto reduce_sum_1 = builder.ReduceSum(relu_1, {1});
  auto program      = builder.Build();

  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(program, target);
  ApplyPass(graph.get(), "OpFusion");

  FLAGS_cinn_fusion_groups_graphviz_dir = "./visualize";
  graph->VisualizeGroupedGraph(graph->groups, {reduce_sum_1->id});
}

TEST(Graph, visualize_recompute) {
  frontend::NetBuilder builder("test");
  auto x              = builder.CreateInput(Float(32), {16, 32}, "x");
  auto y              = builder.CreateInput(Float(32), {32, 16}, "y");
  auto z              = builder.CreateInput(Float(32), {16}, "z");
  auto constant_1     = builder.FillConstant<float>({16}, 1, "constant_1");
  auto add_1          = builder.ElementwiseAdd(z, constant_1);
  auto broadcast_to_1 = builder.BroadcastTo(add_1, {16, 32});
  auto broadcast_to_2 = builder.BroadcastTo(add_1, {32, 16});
  auto add_2          = builder.ElementwiseAdd(x, broadcast_to_1);
  auto add_3          = builder.ElementwiseAdd(y, broadcast_to_2);
  auto program        = builder.Build();

  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(program, target);
  ApplyPass(graph.get(), "OpFusionPass");
  ApplyPass(graph.get(), "FusionMergePass");

  FLAGS_cinn_fusion_groups_graphviz_dir = "./visualize_recompute";
  graph->VisualizeGroupedGraph({add_2->id, add_3->id});
}

class TestAddReduceNode {
 public:
  TestAddReduceNode(Graph* graph)
      : graph_(graph),
        shape_dict_(graph_->GetMutableAttrs<ShapeDict>("infershape")),
        dtype_dict_(graph_->GetMutableAttrs<DtypeDict>("inferdtype")) {}

  NodeData* CreateOutputNode(NodePtr node, const std::string& output_id = "") {
    // create node's output data node
    auto node_id = output_id;
    if (node_id.empty()) {
      node_id = cinn::common::UniqName("var_" + node->id());
    }

    auto graph_node = graph_->RetrieveNode(node_id);
    CHECK(graph_node == nullptr) << "The node " << node->op()->name << "'s output" << node_id
                                 << " had been registered in graph! Please check.";

    auto* output_data = new NodeData(node, 0, 0, node_id);
    node->LinkTo(output_data);
    graph_->RegisterNode(node_id, output_data);

    return output_data;
  }

  std::pair<NodePtr, NodeData*> CreateAllNode(const std::string& node_id) {
    const auto& all_node_id = "all_" + node_id;

    auto all_node = Node::Create(Operator::Get("reduce_sum"), all_node_id, all_node_id);

    int shape_size = shape_dict_[node_id].size();
    std::vector<int> axes(shape_size);
    for (int i = 0; i < shape_size; ++i) {
      axes[i] = i;
    }
    all_node->attrs.attr_store["dim"]      = axes;
    all_node->attrs.attr_store["keep_dim"] = false;

    graph_->RegisterNode(all_node_id, all_node.get());

    // create node's output data node
    auto output_data = CreateOutputNode(all_node);

    shape_dict_[output_data->id()] = framework::shape_t{1};
    shape_dict_[output_data->id()] = framework::shape_t{1};

    dtype_dict_[output_data->id()] = common::Float(32);
    dtype_dict_[output_data->id()] = common::Float(32);

    return std::pair<NodePtr, NodeData*>{all_node, output_data};
  }

  GroupPtr CreateSingleNodeGroup(NodePtr node_ptr) {
    auto node = node_ptr.get();

    // init group
    auto group = std::make_shared<Graph::Group>();

    group->nodes.push_back(node);
    group->nodes_set.insert(node);

    // output nodes
    for (auto& edge : node->outlinks_in_order()) {
      auto output_node_data = edge->sink()->safe_as<NodeData>();
      CHECK(output_node_data) << "Node " << node->id() << "'s output node is nullptr! Please check.";
      for (auto& data_edge : output_node_data->outlinks()) {
        group->output_nodes.insert(data_edge->sink()->safe_as<Node>());
      }
    }

    // input nodes
    for (auto& edge : node->inlinks_in_order()) {
      auto input_node_data = edge->source()->safe_as<NodeData>();
      CHECK(input_node_data) << "Node " << node->id() << "'s input node is nullptr! Please check.";

      // input data has source node
      if (input_node_data->source_node.get()) {
        group->input_nodes[input_node_data->source_node.get()] = 1;
      }
    }

    // group type
    group->op_pattern_kind = framework::kOpaque;

    // use current node as master node for schedule
    group->master_nodes.insert(node);
    group->internal_nodes.insert(node);
    group->group_id = node->id();

    return group;
  }

  std::string Apply(const std::string& node_id) {
    auto graph_node   = graph_->RetrieveNode(node_id);
    auto reduce_nodes = CreateAllNode(node_id);

    graph_node->LinkTo(reduce_nodes.first.get());

    graph_->fusion_groups.emplace_back(CreateSingleNodeGroup(reduce_nodes.first));

    return reduce_nodes.second->id();
  }

 private:
  Graph* graph_;

  ShapeDict& shape_dict_;
  DtypeDict& dtype_dict_;
};

void RunTest(const Target& target, const std::shared_ptr<Graph>& graph, const std::vector<std::string>& input_names) {
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    std::vector<float> vec;
    frontend::InitRandomVector<float>(&vec, tensor->shape().numel(), 0.0f, 1.0f);
    frontend::CopyFromVector<float>(vec, tensor, target);
  }

  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(Graph, add_reduce_node) {
  frontend::NetBuilder builder("test");
  auto x       = builder.CreateInput(Float(32), {242}, "x");
  auto y       = builder.CreateInput(Float(32), {242}, "y");
  auto add_1   = builder.ElementwiseAdd(x, y);
  auto program = builder.Build();

  auto target = common::DefaultTarget();
  auto graph  = std::make_shared<Graph>(program, target);
  ApplyPasses(graph.get(), {"OpFusionPass", "FusionMergePass"});

  CHECK_EQ(graph->fusion_groups.size(), 1UL);

  TestAddReduceNode add_reduce_node(graph.get());
  auto output = add_reduce_node.Apply(add_1->id);

  RunTest(target, graph, {"x", "y"});
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
