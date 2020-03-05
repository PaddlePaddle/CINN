#include "cinn/poly/naive_scheduler.h"

#include <vector>

namespace cinn {
namespace poly {

std::map<std::string, isl::map> NaiveScheduler::BuildSchedule() const {
  std::map<std::string, isl::map> res;

  std::vector<ScheduleGraphNode *> nodes_in_order;
  CHECK(!groups_.empty());
  for (auto &group : groups_) {
    for (auto &node : group.nodes) {
      auto *graph_node = node->As<ScheduleGraphNode>();
      nodes_in_order.push_back(graph_node);
    }
  }

  // order them
  for (int i = 1; i < nodes_in_order.size(); i++) {
    nodes_in_order[i]->time_schedule.OrderAfter(nodes_in_order[i - 1]->time_schedule, 0);
  }

  for (auto *node : nodes_in_order) {
    res[node->id()] = node->time_schedule.to_isl(ctx_);
  }
  return res;
}

void NaiveScheduler::PartitionGroups() {
  // treat each node as a unique group, collect the groups in topological order.
  std::vector<common::GraphNode *> nodes_in_order;
  std::vector<common::GraphEdge *> edges_in_order;
  std::tie(nodes_in_order, edges_in_order) = graph_->topological_order();
  for (auto *node : nodes_in_order) {
    detail::Group group({Shared<poly::DataFlowGraphNode>(node->As<poly::DataFlowGraphNode>())});
    groups_.emplace_back(std::move(group));
  }
}

}  // namespace poly
}  // namespace cinn
