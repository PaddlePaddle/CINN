#include <algorithm>
#include <unordered_set>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::GraphNode;
using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpPatternKind;

auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
absl::flat_hash_map<std::string, framework::shape_t> shape_dict;

struct DomNode {
  GraphNode* ref_node{nullptr};
  DomNode* parent{nullptr};
  OpPatternKind pattern{framework::kOpaque};
  int depth{0};
};

void GetBroadcastPattern(Node* op_node,
                         OpPatternKind* pattern,
                         const absl::flat_hash_map<std::string, framework::shape_t>& shape_dict) {
  if (*pattern == framework::kBroadcast) {
    auto inlinks  = op_node->inlinks();
    auto outlinks = op_node->outlinks();
    CHECK_EQ(inlinks.size(), 2U);
    CHECK_EQ(outlinks.size(), 1U);
    std::vector<framework::shape_t> input_shapes;
    for (auto link : inlinks) {
      auto source = link->source();
      CHECK(shape_dict.count(source->id()));
      input_shapes.push_back(shape_dict.at(source->id()));
    }
    int small_index = input_shapes[0].size() <= input_shapes[1].size() ? 0 : 1;
    auto begin      = std::find(
        input_shapes[1 - small_index].begin(), input_shapes[1 - small_index].end(), input_shapes[small_index][0]);
    bool is_same = true;
    for (int i = 0; i < input_shapes[small_index].size(); i++) {
      if (input_shapes[small_index][i] != (*begin)) {
        is_same = false;
        break;
      } else {
        ++begin;
      }
    }
    if (is_same) {
      *pattern = framework::kElemWise;
    } else {
      LOG(INFO) << "not fuse broadcast";
    }
  }
}

class DomTree {
 public:
  std::vector<DomNode*>& CreatePostDomTree(const std::vector<GraphNode*>& nodes) {
    int size = nodes.size();
    dom_nodes_.resize(nodes.size());
    // construct postdom tree, reverse topological_order
    for (int i = size - 1; i >= 0; i--) {
      auto* dom_node = CreateDomNode(nodes[i]);
      CHECK(dom_node);
      VLOG(2) << "dom_node: " << dom_node->ref_node->id() << ", pattern: " << dom_node->pattern
              << ", depth: " << dom_node->depth;
      if (dom_node->parent) {
        VLOG(2) << dom_node->ref_node->id() << " parent: " << dom_node->parent->ref_node->id();
      }
      dom_nodes_[i] = dom_node;
    }
    return dom_nodes_;
  }

  std::vector<DomNode*> dom_nodes_;

 private:
  OpPatternKind FusePattern(OpPatternKind p0, OpPatternKind p1) { return p0 > p1 ? p0 : p1; }
  DomNode* LCA(DomNode* l, DomNode* r, OpPatternKind* pattern) {
    while (l != r) {
      if (!l || !r) return nullptr;
      if (l->depth < r->depth) {
        *pattern = FusePattern(*pattern, r->pattern);
        r        = r->parent;
      } else if (l->depth > r->depth) {
        *pattern = FusePattern(*pattern, l->pattern);
        l        = l->parent;
      } else {
        *pattern = FusePattern(*pattern, l->pattern);
        *pattern = FusePattern(*pattern, r->pattern);
        l        = l->parent;
        r        = r->parent;
      }
    }
    return l;
  }

  DomNode* FindLCA(GraphNode* graph_node, OpPatternKind* pattern) {
    CHECK(graph_node);
    CHECK(pattern);
    DomNode* parent = nullptr;
    int count       = 0;
    if (graph_node->safe_as<Node>()) {
      auto* node      = graph_node->safe_as<Node>();
      auto& out_links = node->outlinks_in_order(true);
      for (int i = 0; i < out_links.size(); i++) {
        auto sink         = out_links[i]->sink();
        bool has_no_links = sink->outlinks().empty();
        if (i) {
          CHECK(has_no_links) << "only the first out_var of " << node->id() << " links to other op node";
        } else {
          int index = sink->get_index();
          // the first out_var is the parent of the op node
          parent   = dom_nodes_[index];
          *pattern = FusePattern(*pattern, parent->pattern);
          return parent;
        }
      }
    } else {
      auto* node_data = graph_node->safe_as<NodeData>();
      CHECK(node_data);
      auto out_links = node_data->outlinks();
      int count      = 0;
      for (auto link : out_links) {
        auto sink     = link->sink();
        int index     = sink->get_index();
        auto dom_node = dom_nodes_[index];
        if (!count) {
          parent = dom_node;
          CHECK(parent);
        } else {
          // if the out_var links to more than one opnode, then we need to find the LCA
          parent = LCA(parent, dom_node, pattern);
        }
        auto* op_node = sink->safe_as<Node>();
        CHECK(op_node);
        auto op_pattern = op_pattern_dict[op_node->op()];
        VLOG(2) << sink->id() << "'s op pattern is " << op_pattern;
        *pattern = FusePattern(*pattern, op_pattern);
        count++;
      }
      return parent;
    }
  }
  DomNode* CreateDomNode(GraphNode* graph_node) {
    CHECK(graph_node);
    DomNode* dom_node  = new DomNode();
    dom_node->ref_node = graph_node;
    if (graph_node->inlinks().empty() && graph_node->safe_as<NodeData>()) {
      CHECK(graph_node->safe_as<NodeData>());
      // extern input vars
      dom_node->parent  = nullptr;
      dom_node->pattern = framework::kOpaque;
      dom_node->depth   = 0;
    } else {
      OpPatternKind pattern{framework::kElemWise};
      auto* parent      = FindLCA(graph_node, &pattern);
      dom_node->parent  = parent;
      dom_node->pattern = pattern;
      dom_node->depth   = parent ? parent->depth + 1 : 0;
    }
    return dom_node;
  }
};
struct GroupNode {
  GroupNode* parent{nullptr};
  OpPatternKind pattern;
  common::GraphNode* ref_node{nullptr};
  common::GraphNode* master_node{nullptr};
  int index{0};
  int nodes_count{1};
  int op_nodes_count{0};
  // get the root node
  GroupNode* GetRootNode() {
    if (!this->parent) return this;
    GroupNode* root_node = this;
    while (root_node->parent) {
      root_node = root_node->parent;
    }
    // update group node's parent with root_node
    auto* node = this;
    while (node != root_node) {
      auto* parent = node->parent;
      node->parent = root_node;
      node         = parent;
    }
    return root_node;
  }
};
class GraphPartition {
 public:
  std::vector<std::vector<Node*>> Partition(const std::vector<GraphNode*>& graph_nodes,
                                            const std::vector<DomNode*>& dom_nodes) {
    CHECK_EQ(graph_nodes.size(), dom_nodes.size());
    InitGroups(graph_nodes);
    FuseGroups(graph_nodes, dom_nodes);
    SplitGroups(graph_nodes);
#ifdef CINN_WITH_DEBUG
    PrintGroups();
#endif
    return groups_;
  }

 private:
  std::vector<GroupNode*> group_nodes_;
  std::vector<std::vector<Node*>> groups_;
  std::unordered_set<GraphNode*> visited_nodes_;
  void InitGroups(const std::vector<GraphNode*>& graph_nodes) {
    for (int i = 0; i < graph_nodes.size(); i++) {
      GroupNode* group_node = new GroupNode();
      GraphNode* graph_node = graph_nodes[i];
      CHECK(graph_node);
      auto op_node         = graph_node->safe_as<Node>();
      group_node->ref_node = graph_node;
      group_node->index    = graph_node->get_index();
      if (op_node) {
        auto pattern               = op_pattern_dict[op_node->op()];
        group_node->pattern        = pattern;
        group_node->op_nodes_count = 1;
        if (pattern == framework::kOutEWiseFusable) {
          group_node->master_node = graph_node;
        }
      } else {
        // var nodes
        if (graph_node->inlinks().empty()) {
          group_node->pattern = framework::kOpaque;
        } else {
          group_node->pattern = framework::kElemWise;
        }
      }
      group_nodes_.push_back(group_node);
    }
  }
  bool IsSameShape(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    if (shape1.size() != shape2.size()) return false;
    for (int i = 0; i < shape1.size(); i++) {
      if (shape1[i] != shape2[i]) return false;
    }
    return true;
  }
  std::vector<int> GetOutshape(GraphNode* node) {
    CHECK(node);
    auto op_node = node->safe_as<Node>();
    std::vector<int> out_shapes;
    if (op_node) {
      // first out var shape
      CHECK(!op_node->outlinks_in_order().empty());
      auto out_var = op_node->outlinks_in_order().front()->sink();
      CHECK(shape_dict.count(out_var->id()));
      out_shapes = shape_dict[out_var->id()];
    } else {
      CHECK(shape_dict.count(node->id()));
      out_shapes = shape_dict[node->id()];
    }
    return out_shapes;
  }
  bool VerifyOutShape(GraphNode* node1, GraphNode* node2) {
    auto out_shape1 = GetOutshape(node1);
    auto out_shape2 = GetOutshape(node2);
    if (out_shape1.size() == 1 || out_shape2.size() == 1) return true;
    if (out_shape1.size() != out_shape2.size()) return false;
    VLOG(2) << node1->id() << ", out_shape1: " << utils::Join(out_shape1, ", ");
    VLOG(2) << node2->id() << ", out_shape2: " << utils::Join(out_shape2, ", ");
    for (int i = 0; i < out_shape1.size(); i++) {
      if (out_shape1[i] != out_shape2[i]) return false;
    }
    return true;
  }
  template <typename T>
  bool CanFuse(GraphNode* source, GraphNode* sink, T fn) {
    if (visited_nodes_.count(source)) return true;
    visited_nodes_.insert(source);
    auto* group_node = group_nodes_[source->get_index()];
    CHECK(group_node);
    auto* root_node = group_node->GetRootNode();
    CHECK(root_node);
    if (!fn(root_node->pattern, source == sink)) return false;
    if (source == sink) return true;
    auto op_node = source->safe_as<Node>();
    if (op_node) {
      auto& out_links = op_node->outlinks_in_order(true);
      for (int i = 0; i < out_links.size(); i++) {
        auto new_source = out_links[i]->sink();
        // judge only the first out var of the op node can fuse
        if (!i) {
          if (!CanFuse(new_source, sink, fn)) return false;
        } else {
          CHECK(new_source->outlinks().empty()) << "only the first out_var of the op node can link to other op node";
        }
      }
    } else {
      auto& out_links = source->outlinks();
      for (auto link : out_links) {
        auto new_source = link->sink();
        if (!CanFuse(new_source, sink, fn)) return false;
      }
    }
    return true;
  }
  // check all the nodes between source and sink meet the function of fusion.
  template <typename T>
  bool VerifyFuse(GraphNode* source, GraphNode* sink, T fn) {
    auto op_node = source->safe_as<Node>();
    visited_nodes_.clear();
    CHECK(source != sink);
    if (!VerifyOutShape(source, sink)) return false;
    if (op_node) {
      auto& outlinks = op_node->outlinks_in_order(true);
      for (int i = 0; i < outlinks.size(); i++) {
        auto* new_source = outlinks[i]->sink();
        if (!i) {
          // verify all the nodes in the fuse path recursively
          if (!CanFuse(new_source, sink, fn)) return false;
        } else {
          CHECK(new_source->outlinks().empty()) << "only the first out_var of op_node links to other op_node";
        }
      }
    } else {
      auto& outlinks = source->outlinks();
      for (auto link : outlinks) {
        auto* new_source = link->sink();
        // verifyFuse all the nodes in the fuse path recursively
        if (!CanFuse(new_source, sink, fn)) return false;
      }
    }
    return true;
  }
  void MergeNodes(GroupNode* child, GroupNode* parent) {
    child  = child->GetRootNode();
    parent = parent->GetRootNode();
    CHECK(child);
    CHECK(parent);
    if (child == parent) return;
    parent->nodes_count += child->nodes_count;
    parent->op_nodes_count += child->op_nodes_count;
    child->parent = parent;
    if (child->master_node) {
      CHECK(!parent->master_node);
      parent->master_node = child->master_node;
      if (child->pattern > framework::kBroadcast && parent->pattern > framework::kBroadcast) {
        LOG(FATAL) << "can't fuse 2 groups both with complex pattern";
      } else {
        parent->pattern = child->pattern > parent->pattern ? child->pattern : parent->pattern;
      }
    }
  }
  void Fuse(GraphNode* source, GraphNode* sink, GroupNode* target) {
    if (source == sink) return;
    if (visited_nodes_.count(source)) return;
    visited_nodes_.insert(source);
    auto* group_node = group_nodes_[source->get_index()];
    CHECK(group_node);
    MergeNodes(group_node, target);
    auto op_node = source->safe_as<Node>();
    if (op_node) {
      auto& outlinks = op_node->outlinks_in_order(true);
      for (int i = 0; i < outlinks.size(); i++) {
        auto* new_source = outlinks[i]->sink();
        if (!i) {
          Fuse(new_source, sink, target);
        } else {
          CHECK(new_source->outlinks().empty()) << "only the first out_var of op_node links to other op_node";
        }
      }
    } else {
      auto& outlinks = source->outlinks();
      for (auto link : outlinks) {
        auto* new_source = link->sink();
        Fuse(new_source, sink, target);
      }
    }
  }
  void DoFuse(GraphNode* source, GraphNode* sink) {
    auto* group_node = group_nodes_[sink->get_index()];
    CHECK(group_node);
    visited_nodes_.clear();
    CHECK(source != sink);
    Fuse(source, sink, group_node);
  }
  void FuseGroups(const std::vector<GraphNode*>& graph_nodes, const std::vector<DomNode*>& dom_nodes) {
    CHECK_EQ(graph_nodes.size(), dom_nodes.size());
    CHECK_EQ(group_nodes_.size(), dom_nodes.size());
    for (int i = 0; i < graph_nodes.size(); i++) {
      auto* graph_node = graph_nodes[i];
      auto* dom_node   = dom_nodes[i];
      auto* group_node = group_nodes_[i];
      CHECK(graph_node);
      CHECK(dom_node);
      CHECK(group_node);
      if (!dom_node->parent) continue;
      if (group_node->pattern == framework::kOpaque) continue;
      int parent_index       = dom_node->parent->ref_node->get_index();
      auto parent_group_node = group_nodes_[parent_index];
      if (parent_group_node && parent_group_node->GetRootNode() == group_node->GetRootNode()) continue;

      if (group_node->pattern == framework::kOutEWiseFusable) {
        if (dom_node->pattern <= framework::kBroadcast) {
          auto fn       = [](OpPatternKind pattern, bool is_sink) { return pattern <= framework::kBroadcast; };
          auto lca_node = dom_node->parent->ref_node;
          if (VerifyFuse(graph_node, lca_node, fn)) {
            VLOG(2) << "fuse between " << graph_node->id() << " and " << lca_node->id();
            DoFuse(graph_node, lca_node);
          }
        }
      } else if (group_node->pattern <= framework::kBroadcast) {
        if (dom_node->pattern <= framework::kBroadcast) {
          auto fn = [](OpPatternKind pattern, bool is_sink) {
            if (is_sink) {
              return pattern <= framework::kBroadcast || pattern == framework::kOutEWiseFusable;
            } else {
              return pattern <= framework::kBroadcast;
            }
          };
          auto lca_node = dom_node->parent->ref_node;
          if (VerifyFuse(graph_node, lca_node, fn)) {
            VLOG(2) << "fuse between " << graph_node->id() << " and " << lca_node->id();
            DoFuse(graph_node, lca_node);
          }
        }
      }
    }
  }
  void SplitGroups(const std::vector<common::GraphNode*>& graph_nodes) {
    // split groups sorted by topo order
    CHECK_EQ(graph_nodes.size(), group_nodes_.size());
    absl::flat_hash_map<int, std::vector<Node*>> group_maps;
    std::set<int> root_indice;
    for (int i = 0; i < graph_nodes.size(); i++) {
      CHECK(graph_nodes[i]);
      auto* op_node = graph_nodes[i]->safe_as<Node>();
      if (!op_node) continue;
      CHECK(group_nodes_[i]->GetRootNode());
      int root_index = group_nodes_[i]->GetRootNode()->ref_node->get_index();
      group_maps[root_index].push_back(op_node);
      root_indice.insert(root_index);
    }
    for (auto index : root_indice) {
      groups_.push_back(group_maps[index]);
    }
  }
  void PrintGroups() {
    for (int i = 0; i < groups_.size(); i++) {
      VLOG(2) << "group " << i << ": ";
      for (auto& node : groups_[i]) {
        VLOG(2) << node->id() << " ";
      }
    }
  }
};

void OpFusionPass(Graph* graph) {
  shape_dict       = graph->GetMutableAttrs<absl::flat_hash_map<std::string, framework::shape_t>>("infershape");
  auto store_nodes = std::get<0>(graph->topological_order());
  int node_size    = store_nodes.size();
  // construct postdom tree, reverse topological_order
  DomTree tree;
  auto& dom_nodes = tree.CreatePostDomTree(store_nodes);
  // graph partition
  GraphPartition partition;
  graph->groups = partition.Partition(store_nodes, dom_nodes);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(OpFusion) {
  CINN_REGISTER_PASS(OpFusion)
      .describe("This pass traverse the graph and fuse all ops.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPass);

  return true;
}
