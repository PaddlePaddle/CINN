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
 * Schedule a single group with iterator domain considered and follow the stage order.
 */
class PolyGroupScheduler : public SchedulerBase {
 public:
  //! Constructor, this will build a DAG based on the stages.
  explicit PolyGroupScheduler(const std::vector<Stage *> &stages);

  //! Build the schedule, that is set the time schedule following each edge.
  std::vector<Shared<ScheduleGraphNode>> Build();

 private:
  const std::vector<Stage *> &stages_;
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
  explicit PolyScheduler(const std::vector<Stage *> &stages,
                         const std::vector<std::pair<std::string, std::string>> &extra_links = {});

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
