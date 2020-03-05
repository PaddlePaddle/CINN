#pragma once

#include <map>
#include <string>
#include <vector>

#include "cinn/poly/schedule.h"

namespace cinn {
namespace poly {

class NaiveGroupScheduler : public SchedulerBase {
 public:
  //! Constructor, for naive scheduler, each group has just one node.
  explicit NaiveGroupScheduler(Stage *x) {
    AddStage(*x);
    FinishStageAdd();
  }
  //! Just one node, need no schedule.
  void Build() {}
};

/**
 * The NaiveScheduler just schedule each noninlined Tensor as a unique group. Only the `compute_at` will merge two
 * tensor in the same group.
 * It is simple and robust.
 */
class NaiveScheduler : public SchedulerBase {
 public:
  NaiveScheduler() = default;
  explicit NaiveScheduler(const std::vector<Stage *> &stages) {
    for (auto *x : stages) AddStage(*x);
    FinishStageAdd();
  }

  std::unique_ptr<Schedule> BuildSchedule();

 private:
  void PartitionGroups();

 private:
  common::Graph *graph_{};
  std::vector<detail::Group> groups_;
  mutable std::vector<std::string> detailed_dimension_names_;
};

}  // namespace poly
}  // namespace cinn
