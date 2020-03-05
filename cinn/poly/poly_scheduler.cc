#include "cinn/poly/poly_scheduler.h"

#include <glog/logging.h>

#include <deque>

namespace cinn {
namespace poly {

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

std::unique_ptr<Schedule> PolyScheduler::BuildSchedule() {
  std::unique_ptr<Schedule> res(new Schedule);
  for (auto& node : graph_->nodes()) {
    LOG(INFO) << "graph node time_dims: " << node->As<ScheduleGraphNode>() ->time_schedule.space_size();
  }

  // partition the graph to groups.
  PartitionGroups(graph_.get());
  CHECK(!groups_.empty());
  for (auto& node : graph_->nodes()) {
    LOG(INFO) << "graph node time_dims: " << node->As<ScheduleGraphNode>() ->time_schedule.space_size();
  }
  for(auto& group : groups_) {
    for (auto& node : group.nodes) {
      LOG(INFO) << "time_dims " << node->id() << " " << node->As<ScheduleGraphNode>()->time_schedule.space_size();
    }
  }

  // Schedule each group
  ScheduleGroups();

  // Collect result.
  res->groups = groups_;

  for (auto& group : groups_) {
    for (auto& node : group.nodes) {
      LOG(INFO) << "node.id " << node->id();
      res->schedule[node->id()] = node->As<ScheduleGraphNode>()->time_schedule.to_isl(Context::Global().isl_ctx());
    }
  }

  return res;
}

PolyScheduler::PolyScheduler(const std::vector<Stage*>& stages) {
  CHECK_GT(stages.size(), 0) << "No stage is provided";
  graph_ = CreateGraph(stages);
}

void PolyScheduler::PartitionGroups(common::Graph* graph) {
  CHECK(graph);
  CHECK(!graph->nodes().empty());
  groups_ = detail::PartitionGraphByIterationDomain(graph);
}

void PolyScheduler::ScheduleGroup(detail::Group* group) {
  CHECK(group);
  CHECK(!group->nodes.empty());

  // create scheduler for this group.
  std::vector<Stage*> stages;
  for (auto& node : group->nodes) {
    stages.push_back(node->stage.get());
  }

  PolyGroupScheduler scheduler(stages);
  scheduler.Build();
  group->dimension_names = scheduler.detailed_dimension_names();
}

void PolyScheduler::ScheduleGroups() {
  CHECK(!groups_.empty()) << "call PartitionGroups first";
  for (auto& group : groups_) {
    ScheduleGroup(&group);
  }
}

}  // namespace poly
}  // namespace cinn
