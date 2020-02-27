#include "cinn/poly/graph.h"

#include <deque>
#include <map>
#include <set>

namespace cinn {
namespace poly {

const DataFlowGraphNode* DataFlowGraphNode::group_ancestor() const {
  auto* p = this;
  while (p->group_parent) p = p->group_parent;
  return p;
}

DataFlowGraphNode* DataFlowGraphNode::group_ancestor() {
  auto* p = this;
  while (p->group_parent) p = p->group_parent;
  return p;
}

bool DataFlowGraphNode::TransformedDomainIsSame(const DataFlowGraphNode* a, const DataFlowGraphNode* b) {
  VLOG(3) << "a.domain " << a->stage->domain();
  VLOG(3) << "a.transform " << a->stage->transform();
  VLOG(3) << "b.domain " << b->stage->domain();
  VLOG(3) << "b.transform " << b->stage->transform();
  auto a_domain = a->stage->transformed_domain();
  auto b_domain = b->stage->transformed_domain();
  a_domain      = isl::manage(isl_set_set_tuple_name(a_domain.release(), ""));
  b_domain      = isl::manage(isl_set_set_tuple_name(b_domain.release(), ""));
  return isl_set_is_equal(a_domain.get(), b_domain.get());
}

int DataFlowGraphNode::group_height() const {
  int h   = 0;
  auto* p = this;
  while (p) {
    ++h;
    p = p->group_parent;
  }
  return h;
}

DataFlowGraphNode* DataFlowGraphNode::MergeGroup(DataFlowGraphNode* a, DataFlowGraphNode* b) {
  int ah      = a->group_height();
  int bh      = b->group_height();
  auto* a_anc = a->group_ancestor();
  auto* b_anc = b->group_ancestor();
  DataFlowGraphNode* common_anc{};
  if (ah < bh) {  // take a's ancestor
    b_anc->group_parent = a_anc;
    b->group_parent     = a_anc;
    return a_anc;
  } else {
    a_anc->group_parent = b_anc;
    a->group_parent     = b_anc;
    return b_anc;
  }
}
std::string DataFlowGraphNode::id() const {
  // NOTE the stage's id should be unique.
  return stage->id();
}

namespace detail {

//! Visit the nodes in topological order, if one node is valid to visit, visit it and check whether its out link
//! children are ready to visit, merge them to the same group.
std::vector<Group> PartitionGraphByIterationDomain(common::Graph* graph) {
  VLOG(3) << "graph:\n" << graph->Visualize();
  // collect indegree for topological traversal
  std::map<DataFlowGraphNode*, uint16_t> indegree;
  for (common::GraphNode* n : graph->nodes()) {
    auto* node     = n->As<DataFlowGraphNode>();
    indegree[node] = node->inlinks().size();
  }

  std::deque<DataFlowGraphNode*> queue;
  for (auto* n : graph->start_points()) {
    auto* node = n->As<DataFlowGraphNode>();
    queue.push_back(node);
  }
  while (!queue.empty()) {
    auto* node = queue.front();
    queue.pop_front();
    VLOG(4) << "to visit " << node->id();

    for (auto& c : node->outlinks()) {
      auto* child = c->sink()->As<DataFlowGraphNode>();
      VLOG(2) << "  tell child " << c->sink()->id() << " indegree " << indegree[child];
      --indegree[child];

      VLOG(3) << node->stage->transformed_domain() << " -> " << child->stage->transformed_domain();
      if (indegree[child] == 0) {
        if (DataFlowGraphNode::TransformedDomainIsSame(node, child)) {
          VLOG(4) << child->id() << " ready to merge " << node->id() << " with " << child->id();
          DataFlowGraphNode::MergeGroup(node, child);
        }
        queue.push_back(child);
      }
    }
  }

  // gather groups
  std::set<DataFlowGraphNode*> groups_gathered;
  std::vector<DataFlowGraphNode*> groups_in_topo_order;

  std::vector<common::GraphNode*> nodes_in_order;
  std::vector<common::GraphEdge*> edges_in_order;
  std::map<DataFlowGraphNode*, std::vector<DataFlowGraphNode*>> node_groups;
  std::tie(nodes_in_order, edges_in_order) = graph->topological_order();
  for (auto* n : nodes_in_order) {
    auto* node     = n->As<DataFlowGraphNode>();
    auto* ancestor = node->group_ancestor();
    if (!groups_gathered.count(ancestor)) {
      groups_gathered.insert(ancestor);
      groups_in_topo_order.push_back(ancestor);
    }

    node_groups[ancestor].push_back(node);
  }

  std::vector<Group> groups;
  // preparing result
  for (auto* ancestor : groups_in_topo_order) {
    Group group;
    for (auto* c : node_groups[ancestor]) {
      group.nodes.push_back(c);
    }
    groups.emplace_back(group);
  }

  // NOTE DEBUG
  // check there are same count of nodes both in the orginal graph and the groups.
  // @{
  int num_node_in_groups = 0;
  for (auto& group : groups) num_node_in_groups += group.nodes.size();
  CHECK_EQ(num_node_in_groups, graph->num_nodes());
  // @}

  return groups;
}

}  // namespace detail

std::unique_ptr<common::Graph> CreateGraph(const std::vector<Stage*>& stages) {
  std::map<std::string, Shared<DataFlowGraphNode>> id2stage;
  for (auto* x : stages) id2stage[x->id()] = make_shared<DataFlowGraphNode>(x);

  for (auto* stage : stages) {
    auto depend_statement_names = stage->input_statements();
    VLOG(3) << stage->id() << " depend " << utils::Join(depend_statement_names, ", ");
    for (auto& depend_statement : depend_statement_names) {
      auto input_it = id2stage.find(depend_statement);
      // We removed some node in the original stages(such as placeholders), so that there might be missing of some input
      // nodes, just ignore the dependence.
      if (input_it != std::end(id2stage)) {
        auto& input_node = id2stage.at(depend_statement);
        input_node->LinkTo(id2stage.at(stage->id()).get());
      }
    }
  }

  std::unique_ptr<common::Graph> graph(new common::Graph);
  for (auto& item : id2stage) graph->RegisterNode(item.first, item.second.get());
  VLOG(3) << "created graph:\n" << graph->Visualize();
  return graph;
}

}  // namespace poly
}  // namespace cinn
