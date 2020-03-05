#pragma once
/**
 * This file defines Schedule related concepts.
 */
#include <algorithm>
#include <map>
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
#include "cinn/poly/stage.h"

namespace cinn {
namespace poly {

/**
 * The dimension with time space.
 */
struct TimeDim {
  //! time of this dimension.
  int time;
  //! name of this dimension.
  std::string dim;

  TimeDim() = default;
  TimeDim(std::string dim, int time) : dim(std::move(dim)), time(time) {}
};

class ScheduleGraphNode;
struct ScheduleGraph : public common::Graph {};

/**
 * ISL schedule map with time space, used to generate the final schedule.
 * The map it generates is like { [x,y] -> [t0,x,t1,y] }, the t0 and t1 are time space.
 */
struct TimeSchedule {
  TimeSchedule(const std::string &id, const std::vector<std::string> &dims);

  void ResizeTimeSpace(int size) {
    CHECK_LE(size, kMaxDims);
    time_dims_.resize(size);
  }

  //! Schedule this after \p other in \p level.
  void OrderAfter(const TimeSchedule &other, int level);

  size_t space_size() const { return time_dims_.size(); }

  const std::string &id() const;

  //! Get the isl map.
  isl::map to_isl(isl::ctx ctx) const;

  //! ISL range format, such as '[dup, t0, t1]: dup=0 and t0=0 and t1=i]'
  std::string __str__() const;

  //! Get the axis names with the original dimension names and faked time dimensions.
  std::vector<std::string> final_axis_names() const;

  std::vector<std::string> domain_dims;
  int duplicate_id{};

  constexpr static int kMaxDims = 50;

 private:
  std::vector<TimeDim> time_dims_;
  std::string id_;
};

/**
 * The base class for all the Scheduler, it helps to schedule the nodes in a group(isl space).
 */
class SchedulerBase {
 public:
  /**
   * Wrap the iterator names with time space fake names, it is used for isl AST to set iterator names.
   * @param names the original iterator names.
   * @return the iterator names with time space included.
   */
  static std::vector<std::string> WrapIteratorNames(const std::vector<std::string> &names);

  /**
   * Mark this should schedule after another.
   *
   * @param b
   * @param level
   */
  SchedulerBase &After(const Stage &a, const Stage &b, int level);
  /**
   * Mark this should schedule before another.
   * @param b
   * @param level
   */
  SchedulerBase &Before(const Stage &a, const Stage &b, int level);

  const std::vector<std::string> &detailed_dimension_names() const { return detailed_dimension_names_; }

 protected:
  /**
   * Register an Element to the scheduler.
   */
  void AddStage(const Stage &x);

  /**
   * Finalize the registration.
   */
  void FinishStageAdd();

  /**
   * Tell whether the registration is finalized.
   */
  bool finalized() const { return registration_finalized_; }
  int space_size() const { return space_size_; }

 protected:
  /**
   * The polyhedral schedule, any schedule is performed on it.
   * We use the time-space map to record the schedule information, the format is borrowed from Tiramisu project:
   * [time,dim,time,dim,time,dim ...]
   */
  int space_size_{0};
  mutable isl::ctx ctx_{Context::Global().isl_ctx()};
  mutable ScheduleGraph schedule_graph_;
  // Record the longest dimensions(of some stage) to be the final detailed dimension names. It might be used for ISL AST
  // to set iterator names and generate readable code.
  mutable std::vector<std::string> detailed_dimension_names_;

 private:
  bool registration_finalized_{false};
};

/**
 * A container type to contain the schedule information of a graph(several groups).
 */
struct Schedule {
  //! The groups partitioned from the dependency graph.
  std::vector<detail::Group> groups;
  //! id to the isl schedule for each node.
  std::map<std::string, isl::map> schedule;
};

/**
 * Schedule Kind.
 */
enum class ScheduleKind {
  //! Basic strategy, each status is scheduled seperately.
  Naive = 0,
  //! The strategy with iteration domain considered.
  Poly = 1,
};

std::unique_ptr<Schedule> CreateSchedule(const ir::Tensor &tensor, ScheduleKind schedule_kind = ScheduleKind::Poly);
std::unique_ptr<Schedule> CreateSchedule(const std::vector<Stage *> &stages,
                                         ScheduleKind schedule_kind = ScheduleKind::Poly);

/**
 * Gather the stages in the input tensors and their dependencies
 * @param xs The input tensors.
 * @param with_placeholder Whether to include placeholders(default false).
 * @returns The stages in topological order follow the connection to `xs`.
 */
std::vector<Stage *> GatherStagesInTensors(const std::vector<ir::Tensor> &xs, bool with_placeholder = false);

struct ScheduleGraphEdge : public common::GraphEdge {
  ScheduleGraphEdge(common::GraphNode *a, common::GraphNode *b) : common::GraphEdge(a, b) {}

  //! Dependency level.
  int level;
};

/**
 * Node in the schedule graph.
 */
struct ScheduleGraphNode : public DataFlowGraphNode {
  TimeSchedule time_schedule;

  //! NOTE this id is not human-readable.
  std::string id() const override { return std::to_string(reinterpret_cast<size_t>(this)); }

  explicit ScheduleGraphNode(const std::string &id, const std::vector<std::string> &dims) : time_schedule(id, dims) {}
};

std::map<std::string, isl::map> CollectSchedleMapFromGroup(const detail::Group &group);

}  // namespace poly
}  // namespace cinn
