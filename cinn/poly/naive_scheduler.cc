#include "cinn/poly/naive_scheduler.h"

#include <vector>

namespace cinn {
namespace poly {

std::unique_ptr<Schedule> NaiveScheduler::BuildSchedule() {
  PartitionGroups();
  CHECK(!groups_.empty());

  for (auto &group : groups_) {
    std::vector<Stage *> status;
    CHECK_EQ(group.nodes.size(), 1UL);
    NaiveGroupScheduler scheduler(const_cast<Stage *>(group.nodes.front()->stage));
    scheduler.Build();
  }

  std::unique_ptr<Schedule> res(new Schedule);
  res->groups = groups_;

  return res;
}

void NaiveScheduler::PartitionGroups() {
  // treat each node as a unique group, collect the groups in topological order.
  auto topo_order = schedule_graph_.topological_order();  // NOLINT
  auto &nodes_in_order = std::get<0>(topo_order);
  auto &edges_in_order = std::get<1>(topo_order);

  for (auto *node : nodes_in_order) {
    ScheduleGroup group;
    group.nodes.push_back(node->safe_as<ScheduleGraphNode>());
    groups_.emplace_back(std::move(group));
  }
}

}  // namespace poly
}  // namespace cinn
