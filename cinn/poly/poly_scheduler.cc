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
  // collect indegrees for topological traversal
  std::map<DataFlowGraphNode*, uint16_t> indegree;
  for (common::GraphNode* n : graph->nodes()) {
    auto* node     = n->As<DataFlowGraphNode>();
    indegree[node] = node->inlinks().size();
  }

  std::map<std::string, DataFlowGraphNode*> name2node;
  for (auto* n : graph->nodes()) {
    name2node[n->id()] = n->As<DataFlowGraphNode>();
  }

  // topological sort.
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
        // Merge the two groups if their iteration domain is the same.
        if (DataFlowGraphNode::TransformedDomainIsSame(node, child)) {
          VLOG(4) << child->id() << " ready to merge " << node->id() << " with " << child->id();
          DataFlowGraphNode::MergeGroup(node, child);
        }
        queue.push_back(child);
      }
    }
  }

  // process the ComputeAt relation.
  for (auto* n : graph->nodes()) {
    auto* node = n->As<DataFlowGraphNode>();
    for (auto& compute_at : node->stage->compute_ats()) {
      CHECK(compute_at.IsCompatible(node->stage.get())) << "The registered ComputeAt is not compatible";
      // check the endpoints of compute_at has data dependency.
      auto* node0 = node;
      auto* node1 = name2node[compute_at.stage->id()];
      CHECK(node0->IsLinkedTo(node1));

      node0->group_parent = node1->group_parent ? node1->group_parent : node1;

      // TODO(Superjomn) Consider the case node1 is a parent.
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

  // partition the DataFlowGraph to groups.
  auto dfg_groups = PartitionGroups(dfg_.get());
  CHECK(!dfg_groups.empty());

  // transform the DFG groups to schedule groups.
  CHECK(!schedule_graph_.nodes().empty());
  CHECK_EQ(schedule_graph_.nodes().size(), dfg_->nodes().size()) << "DFG graph is not match schedule graph";
  schedule_groups_.clear();
  for (auto& dfg_group : dfg_groups) {
    ScheduleGroup group;
    for (auto& node : dfg_group.nodes) {
      auto* schedule_node = schedule_graph_.RetriveNode(node->id());
      CHECK(schedule_node) << "missing node " << node->id() << " in schedule graph";
      group.nodes.push_back(schedule_node->As<ScheduleGraphNode>());
    }
    schedule_groups_.emplace_back(std::move(group));
  }
  CHECK_EQ(schedule_groups_.size(), dfg_groups.size());

  // Schedule each group
  ScheduleGroups();

  // Collect result.
  res->groups = schedule_groups_;

  for (auto& group : schedule_groups_) {
    for (auto& node : group.nodes) {
      res->schedule[node->id()] = node->time_schedule.to_isl(Context::Global().isl_ctx());
    }
  }

  return res;
}

PolyScheduler::PolyScheduler(const std::vector<Stage*>& stages) {
  CHECK_GT(stages.size(), 0) << "No stage is provided";

  dfg_ = CreateGraph(stages);

  for (auto* stage : stages) {
    AddStage(*stage);
  }
  FinishStageAdd();
}

std::vector<detail::Group> PolyScheduler::PartitionGroups(DataFlowGraph* graph) {
  CHECK(graph);
  CHECK(!graph->nodes().empty());
  return detail::PartitionGraphByIterationDomain(graph);
}

void PolyScheduler::ScheduleAGroup(ScheduleGroup* group) {
  CHECK(group);
  CHECK(!group->nodes.empty());

  // create scheduler for this group.
  std::vector<Stage*> stages;
  for (auto& node : group->nodes) {
    stages.push_back(const_cast<Stage*>(node->stage));
  }

  PolyGroupScheduler scheduler(stages);
  scheduler.Build();
  group->dimension_names = scheduler.detailed_dimension_names();
}

void PolyScheduler::ScheduleGroups() {
  CHECK(!schedule_groups_.empty()) << "call PartitionGroups first";
  for (auto& group : schedule_groups_) {
    ScheduleAGroup(&group);
  }
}

}  // namespace poly
}  // namespace cinn
