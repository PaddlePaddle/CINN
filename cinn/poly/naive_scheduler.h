#pragma once

#include <map>
#include <string>
#include <vector>

#include "cinn/poly/schedule.h"

namespace cinn {
namespace poly {

/**
 * The NaiveScheduler just schedule each noninlined Tensor as a unique group. Only the `compute_at` will make two tensor
 * in the same group. It is simple and robust.
 */
class NaiveScheduler : public SchedulerBase {
 public:
  NaiveScheduler() = default;
  explicit NaiveScheduler(const std::vector<Stage *> &stages);

  std::map<std::string, isl::map> BuildSchedule() const;

 private:
  void PartitionGroups();

 private:
  common::Graph *graph_{};
  std::vector<detail::Group> groups_;
  mutable std::vector<std::string> detailed_dimension_names_;
};

}  // namespace poly
}  // namespace cinn
