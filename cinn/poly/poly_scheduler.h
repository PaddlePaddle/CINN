#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/lang/tensor.h"
#include "cinn/poly/graph.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/map.h"
#include "cinn/poly/schedule.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace poly {

/**
 * Schedule a single group with iterator domain considered.
 */
class PolyGroupScheduler : public SchedulerBase {
 public:
  //! Constructor, this will build a DAG based on the stages.
  explicit PolyGroupScheduler(const std::vector<Stage *> &stages) {
    CHECK_GT(stages.size(), 0) << "No stage is provided";
    for (auto *stage : stages) {
      AddStage(*stage);
    }
    FinishStageAdd();
  }

  //! Build the schedule, that is set the time schedule following each edge.
  void Build() {
    ScheduleGraph::node_order_t node_order;
    ScheduleGraph::edge_order_t edge_order;
    CHECK(!schedule_graph_.nodes().empty());
    std::tie(node_order, edge_order) = schedule_graph_.topological_order();
    for (auto *edge : edge_order) {
      auto *schedule_edge = edge->as<ScheduleGraphEdge>();
      auto *a_node        = schedule_graph_.RetriveNode(edge->source()->As<ScheduleGraphNode>()->time_schedule.id())
                         ->As<ScheduleGraphNode>();
      auto *b_node = schedule_graph_.RetriveNode(edge->sink()->As<ScheduleGraphNode>()->time_schedule.id())
                         ->As<ScheduleGraphNode>();
      CHECK(a_node);
      CHECK(b_node);

      int level = schedule_edge->level;
      b_node->time_schedule.OrderAfter(a_node->time_schedule, level);
    }
  }
};

/**
 * PolyScheduler - Perform schedule on polyhedral model.
 * It takes a normal schedule as input, merge two stages automatically if they have the same domain.
 */
class PolyScheduler : public SchedulerBase {
 public:
  /**
   * Constructor.
   * @param schedule A normal isl schedule, such as '{ S[i,j] -> [i,j] }'
   *
   * The schedule input can be transformed, that's ok, such as
   *   '{ S[i,j] -> [i_outer, i_inner, j]: i_outer=floor(i/4) and i_inner=i%4 }'
   * that's OK.
   */
  explicit PolyScheduler(const std::vector<Stage *> &stages);

  /**
   * Build and create schedule.
   */
  std::unique_ptr<Schedule> BuildSchedule();

 private:
  //! Partition the graph into several groups.
  std::vector<detail::Group> PartitionGroups(DataFlowGraph *graph);
  //! Schedule a single group.
  void ScheduleAGroup(ScheduleGroup *group);
  //! Schedule all the groups.
  void ScheduleGroups();

  std::unique_ptr<DataFlowGraph> dfg_;

  //! The groups of ScheduleNode groups.
  std::vector<ScheduleGroup> schedule_groups_;
};

}  // namespace poly
}  // namespace cinn
