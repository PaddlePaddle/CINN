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

#include "cinn/hlir/pass/dot_merger.h"

#include <fstream>

#include "cinn/common/graph_utils.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/pass.h"

namespace cinn {
namespace hlir {
namespace pass {
namespace {

using common::GraphNode;
using framework::Node;
using framework::NodeData;
using framework::Operator;

template <typename T>
using OpValueType  = cinn::hlir::framework::OpValueType<T>;
using infershape_t = std::function<std::vector<framework::shape_t>(const std::vector<framework::shape_t>&,
                                                                   const framework::AttrMapType&)>;
using inferdtype_t = std::function<std::vector<Type>(const std::vector<Type>&, const framework::AttrMapType&)>;
using dtype_dict_t = absl::flat_hash_map<std::string, common::Type>;
using shape_dict_t = absl::flat_hash_map<std::string, framework::shape_t>;

bool accessible(GraphNode* start, GraphNode* end) {
  std::set<GraphNode const*> marked;
  std::function<void(GraphNode const*)> dfs = [&](GraphNode const* node) {
    marked.emplace(node);
    for (const auto& edge : node->outlinks()) {
      if (!marked.count(edge->sink())) {
        dfs(edge->sink());
      }
    }
  };
  dfs(start);
  return marked.count(end);
}

template <typename T>
T get_attr(Node* instr, const std::string& attr, T def) {
  if (!instr->attrs.attr_store.count(attr)) {
    LOG(INFO) << "not find attr: " << attr;
    return def;
  }
  LOG(INFO) << "found attr: " << attr << " = " << absl::get<T>(instr->attrs.attr_store.at(attr));
  return absl::get<T>(instr->attrs.attr_store.at(attr));
}

NodeData* input_operand(Node* instr, int idx) { return instr->inlinks_in_order()[idx]->source()->safe_as<NodeData>(); }
NodeData* output_operand(Node* instr, int idx) { return instr->outlinks_in_order()[idx]->sink()->safe_as<NodeData>(); }

void remove_node(framework::Graph* graph, GraphNode* node) {
  auto inlinks = node->inlinks();
  for (auto& link : inlinks) {
    link->source()->UnLinkSingleTo(link->sink());
  }
  auto outlinks = node->outlinks();
  for (auto& link : outlinks) {
    link->source()->UnLinkSingleTo(link->sink());
  }
  graph->DropNode(node);
}

template <typename T>
bool all_equal(const T& arg) {
  return arg[0] == arg[1];
}

template <typename T, typename... Args>
bool all_equal(const T& arg, const Args&... args) {
  return all_equal(arg) && all_equal(args...);
}

void InferShape(Node* node, dtype_dict_t& dtype_dict, shape_dict_t& shape_dict) {
  auto op_infershape = Operator::GetAttrs<infershape_t>("infershape");
  auto op_inferdtype = Operator::GetAttrs<inferdtype_t>("inferdtype");
  CHECK(node) << "The node can not be nullptr.";

  auto product = [](const framework::shape_t& shape) {
    framework::dim_t numel = 1;
    std::for_each(shape.begin(), shape.end(), [&numel](framework::dim_t dim) { numel *= dim; });
    return numel;
  };

  std::vector<framework::shape_t> inputs_shape;
  std::vector<Type> inputs_dtype;
  for (auto& in_edge : node->inlinks_in_order()) {
    auto* source_node = in_edge->source()->safe_as<NodeData>();
    CHECK(source_node);
    CHECK(shape_dict.count(source_node->id())) << "No shape for " << source_node->id();
    CHECK(dtype_dict.count(source_node->id())) << "No dtype for " << source_node->id();
    inputs_shape.push_back(shape_dict[source_node->id()]);
    inputs_dtype.push_back(dtype_dict[source_node->id()]);

    CHECK(product(inputs_shape.back())) << node->id() << " 's Input Node " << source_node->id() << "["
                                        << utils::Join(inputs_shape.back(), ",")
                                        << "]'s size should not zero ! Please check.";
  }

  auto out_shape = op_infershape[node->op()](inputs_shape, node->attrs.attr_store);
  auto out_dtype = op_inferdtype[node->op()](inputs_dtype, node->attrs.attr_store);

  CHECK_GE(node->outlinks_in_order().size(), out_shape.size())
      << "The output number of node " << node->id() << " is " << node->outlinks_in_order().size()
      << " , which is smaller than the output shape size " << out_shape.size() << " . And the op type is "
      << node->op()->name;
  CHECK_GE(node->outlinks_in_order().size(), out_dtype.size())
      << "The output number of node " << node->id() << " is " << node->outlinks_in_order().size()
      << " , which is smaller than the output dtype size " << out_dtype.size() << " . And the op type is "
      << node->op()->name;

  int counter = 0;
  for (auto& out_edge : node->outlinks_in_order()) {
    auto* sink_node = out_edge->sink()->safe_as<NodeData>();
    CHECK(sink_node);

    VLOG(3) << "Infershape: " << sink_node->id() << " " << utils::Join(out_shape[counter], ",");
    shape_dict[sink_node->id()] = out_shape[counter];
    dtype_dict[sink_node->id()] = out_dtype[counter];

    CHECK(product(out_shape[counter])) << node->id() << " 's Output Node " << sink_node->id() << "["
                                       << utils::Join(out_shape[counter], ",")
                                       << "]'s size should not zero ! Please check.";

    counter++;
  }
}

class DotBuilder {
 public:
  explicit DotBuilder(framework::Graph* graph)
      : graph_{graph},
        dtype_dict_{graph_->GetMutableAttrs<dtype_dict_t>("inferdtype")},
        shape_dict_{graph_->GetMutableAttrs<shape_dict_t>("infershape")} {}

  framework::Graph* graph() const { return graph_; }
  const dtype_dict_t& dtype_dict() const { return dtype_dict_; };
  const shape_dict_t& shape_dict() const { return shape_dict_; };

  NodeData* Var(common::Shared<Node>& producer) {
    LOG(INFO) << "DotBuilder::Var";
    auto* res = new NodeData(producer, 0, 0, "var___dot_merger_" + std::to_string(idx_++), false);
    graph_->RegisterNode(producer->id(), res);
    graph_->RegisterNode(res->id(), producer.get());
    producer->LinkTo(res);
    InferShape(producer.get(), dtype_dict_, shape_dict_);
    return res;
  }

  NodeData* Concat(int axis, std::vector<NodeData*> inputs) {
    LOG(INFO) << "DotBuilder::Concat";
    const std::string type{"concat"};
    LOG(INFO) << "DotBuilder::Concat";
    Node* tmp = new Node(
        framework::Operator::Get(type), type + "__dot_merger_", type + "__dot_merger_" + std::to_string(idx_++));
    LOG(INFO) << "DotBuilder::Concat";
    common::Shared<Node> instr(tmp);
    LOG(INFO) << "DotBuilder::Concat";
    // auto instr                      = std::make_shared<Node>(framework::Operator::Get(type), gen_name(type));
    instr->attrs.attr_store["axis"] = axis;
    for (auto* in : inputs) {
      in->LinkTo(instr.get());
    }
    auto* output = Var(instr);
    return output;
  }

  NodeData* Matmul(bool trans_a, bool trans_b, bool trans_out, float alpha, NodeData* lhs, NodeData* rhs) {
    LOG(INFO) << "DotBuilder::Matmul";
    const std::string type{"matmul"};
    auto instr                           = common::Shared<Node>(new Node(
        framework::Operator::Get(type), type + "__dot_merger_", type + "__dot_merger_" + std::to_string(idx_++)));
    matmul_                              = instr.get();
    instr->attrs.attr_store["trans_a"]   = trans_a;
    instr->attrs.attr_store["trans_b"]   = trans_b;
    instr->attrs.attr_store["trans_out"] = trans_out;
    instr->attrs.attr_store["alpha"]     = alpha;
    lhs->LinkTo(instr.get());
    rhs->LinkTo(instr.get());
    auto* output = Var(instr);
    return output;
  }

  NodeData* Slice(
      std::vector<int> axes, std::vector<int> starts, std::vector<int> ends, NodeData* input, NodeData* output) {
    LOG(INFO) << "DotBuilder::Slice";
    const std::string type{"slice"};
    common::Shared<Node> instr(new Node(
        framework::Operator::Get(type), type + "__dot_merger_", type + "__dot_merger_" + std::to_string(idx_++)));
    // auto instr                             = std::make_shared<Node>(framework::Operator::Get(type), gen_name(type));
    instr->attrs.attr_store["axes"]        = std::move(axes);
    instr->attrs.attr_store["starts"]      = std::move(starts);
    instr->attrs.attr_store["ends"]        = std::move(ends);
    instr->attrs.attr_store["infer_flags"] = std::vector<int>{};
    instr->attrs.attr_store["strides"]     = std::vector<int>{};
    input->LinkTo(instr.get());
    instr->LinkTo(output);
    graph_->RegisterNode(instr->id(), instr.get());
    InferShape(instr.get(), dtype_dict_, shape_dict_);
    cache_.emplace_back(output->source_node);
    output->source_node = instr;
    return output;
  }

  Node* matmul_op() const { return matmul_; }

 private:
  static int idx_;
  framework::Graph* graph_;
  dtype_dict_t& dtype_dict_;
  shape_dict_t& shape_dict_;
  Node* matmul_{};
  std::vector<framework::NodePtr> cache_;
};

int DotBuilder::idx_;
// std::vector<framework::NodePtr> DotBuilder::cache_;

class DotMergerPass {
 public:
  void Apply(framework::Graph* graph) {
    auto clusters = GetClusters(graph, "matmul");
    std::set<Node*> nodes_to_remove;
    DotBuilder builder(graph);
    for (auto& c : clusters) {
      LOG(INFO) << "cluster: " << c.first->id() << ", size = " << c.second.size();
      auto& dots = c.second;
      for (size_t i = 0; i < dots.size(); ++i) {
        auto*& a = dots[i];
        if (!a) {
          LOG(INFO) << "a is null!";
          continue;
        }
        for (size_t j = i + 1; j < dots.size(); ++j) {
          auto* b = dots[j];
          LOG(INFO) << "i =" << i << ", j = " << j << "(" << a->id() << "," << b->id() << ")";
          if (!b) {
            LOG(INFO) << "b is null!";
            continue;
          }
          if (nodes_to_remove.count(a) || nodes_to_remove.count(b)) {
            LOG(INFO) << "remove!!";
            continue;
          }
          if (accessible(a, b)) {
            LOG(INFO) << "1 access : " << a->id() << ", " << b->id();
            continue;
          }
          if (accessible(b, a)) {
            LOG(INFO) << "2 access : " << a->id() << ", " << b->id();
            continue;
          }
          LOG(INFO) << "fuse: " << a->id() << " (" << a << "), " << b->id() << " (" << b << "), ";
          auto* merged = MergeDots(&builder, a, b);
          if (merged) {
            LOG(INFO) << "--- nodes_to_remove: " << a << ", " << b;
            LOG(INFO) << "--- nodes_to_remove: " << a->id() << " (" << a << "), " << b->id() << " (" << b << "), ";
            nodes_to_remove.insert(a);
            nodes_to_remove.insert(b);
            dots[i] = merged;
            dots[j] = nullptr;
          }
        }
      }
    }
    for (auto* n : nodes_to_remove) {
      remove_node(graph, n);
    }
  }

 private:
  static std::map<NodeData*, std::vector<Node*>> GetClusters(framework::Graph* graph, const std::string& op_type) {
    std::map<NodeData*, std::vector<Node*>> clusters;
    auto nodes = std::get<0>(graph->topological_order());
    for (auto* n : nodes) {
      auto* op_node = n->safe_as<Node>();
      if (op_node && op_node->op()->name == op_type) {
        for (auto& edge : n->inlinks()) {
          auto* var_node = edge->source()->safe_as<NodeData>();
          CHECK(var_node) << "The variable node can not be null.";
          clusters[var_node].push_back(op_node);
        }
      }
    }
    std::vector<std::map<NodeData*, std::vector<Node*>>::iterator> del;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
      if (it->second.size() < 2) {
        del.push_back(it);
      }
    }
    for (auto& it : del) {
      clusters.erase(it);
    }
    return clusters;
  }

  static Node* MergeDots(DotBuilder* builder, Node* a, Node* b) {
    CHECK(a && b) << "The pointer of node is illegal.";
    const std::array<bool, 2> trans_a{get_attr<bool>(a, "trans_a", false), get_attr<bool>(b, "trans_a", false)};
    const std::array<bool, 2> trans_b{get_attr<bool>(a, "trans_b", false), get_attr<bool>(b, "trans_b", false)};
    // const std::array<bool, 2> trans_out{get_attr<bool>(a, "trans_out"), get_attr<bool>(b, "trans_out")};
    const std::array<float, 2> alpha{get_attr<float>(a, "alpha", 1.f), get_attr<float>(b, "alpha", 1.f)};
    if (!all_equal(trans_a, trans_b, alpha)) {
      return nullptr;
    }
    bool lhs{true};
    int axis{1};
    NodeData *shared_input{}, *input_a{}, *input_b{};
    if (input_operand(a, 1) == input_operand(b, 1)) {
      LOG(INFO) << "input_operand(a, 1) == input_operand(b, 1)";
      shared_input = input_operand(a, 1);
      input_a      = input_operand(a, 0);
      input_b      = input_operand(b, 0);
      lhs          = false;
      if (!trans_a[0]) {
        axis = 0;
      } else if (trans_b[0]) {
        axis = 0;
      }
    } else if (input_operand(a, 0) == input_operand(b, 0)) {
      LOG(INFO) << "input_operand(a, 0) == input_operand(b, 0)";
      shared_input = input_operand(a, 0);
      input_a      = input_operand(a, 1);
      input_b      = input_operand(b, 1);
    } else {
      LOG(INFO) << "not fuse!!";
      return nullptr;
    }
    LOG(INFO) << shared_input->id() << ", " << input_a->id() << ", " << input_b->id() << ", axis=" << axis;
    CHECK(shared_input && input_a && input_b) << "The input node type must be variable.";
    // CHECK(builder->shape_dict().count(input_a->id())) << "not found " << input_a->id();
    auto shape_shared = builder->shape_dict().at(shared_input->id());
    auto shape_a      = builder->shape_dict().at(input_a->id());
    // CHECK(builder->shape_dict().count(input_b->id())) << "not found " << input_b->id();
    auto shape_b = builder->shape_dict().at(input_b->id());
    LOG(INFO) << "shape_shared: " << shape_shared[0] << "," << shape_shared[1];
    LOG(INFO) << "shape_a : " << shape_a[0] << ", " << shape_a[1] << ";" << shape_b[0] << ", " << shape_b[1];
    CHECK_EQ(shape_a[1 - axis], shape_b[1 - axis])
        << "The shape of matmul is error. " << shape_a.size() << ", " << shape_b.size();
    auto* concat_out = builder->Concat(axis, {input_a, input_b});
    NodeData* matmul_out{};
    if (!lhs) {
      matmul_out = builder->Matmul(trans_a[0], trans_b[0], false, alpha[0], concat_out, shared_input);
    } else {
      matmul_out = builder->Matmul(trans_a[0], trans_b[0], false, alpha[0], shared_input, concat_out);
    }
    auto* output_a = output_operand(a, 0);
    auto* output_b = output_operand(b, 0);
    builder->Slice({axis}, {0}, {shape_a[axis]}, matmul_out, output_a);
    builder->Slice({axis}, {shape_a[axis]}, {shape_a[axis] + shape_b[axis]}, matmul_out, output_b);
    return builder->matmul_op();
  }
};

}  // namespace

void DotMergerPassFunc(framework::Graph* graph) {
  {
    std::string str = graph->Visualize();
    std::stringstream ss;
    ss << str;
    std::string myString = ss.str();

    std::ofstream file("before.txt", std::ofstream::out | std::ofstream::trunc);
    file << myString;
  }

  DotMergerPass pass;
  pass.Apply(graph);

  {
    std::string str = graph->Visualize();
    std::stringstream ss;
    ss << str;
    std::string myString = ss.str();

    std::ofstream file("after.txt", std::ofstream::out | std::ofstream::trunc);
    file << myString;
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(DotMerger) {
  CINN_REGISTER_PASS(DotMerger)
      .describe("")
      .set_change_structure(false)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::DotMergerPassFunc);
  return true;
}
