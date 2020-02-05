#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/poly/element.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/map.h"

namespace cinn {
namespace poly {

struct TimeDim {
  //! time of this dimension.
  int time;
  //! name of this dimension.
  std::string dim;

  TimeDim() = default;
  TimeDim(std::string dim, int time) : dim(std::move(dim)), time(time) {}
};

struct DependFlow {
  //! Map from the depended Element.id to the level.
  std::unordered_map<std::string, int> depend_level;
};

class ScheduleGraphNode;
struct ScheduleGraph : public common::Graph {};

/**
 * The range of the schedule.
 */
struct TimeSchedule {
  //! ISL range format, such as '[dup, t0, t1]: dup=0 and t0=0 and t1=i]'
  std::string __str__() const;

  TimeSchedule(const std::vector<std::string> &dims) {
    domain_dims = dims;
    for (auto &dim : domain_dims) {
      time_dims.emplace_back(dim, 0);
    }
  }

  void ResizeTimeSpace(int size) { time_dims.resize(size); }

  //! Get the isl map.
  isl::map to_isl(isl::ctx ctx) const { return isl::map(ctx, __str__()); }

  std::string id;
  std::vector<std::string> domain_dims;
  int duplicate_id{};
  std::vector<TimeDim> time_dims;
};

/**
 * Scheduler - Perform schedule on polyhedral model.
 * It takes a normal schedule as input, and transform it into
 *
 */
class Scheduler {
 public:
  /**
   * Constructor.
   * @param schedule A normal isl schedule, such as '{ S[i,j] -> [i,j] }'
   *
   * The schedule input can be transformed, that's ok, such as
   *   '{ S[i,j] -> [i_outer, i_inner, j]: i_outer=floor(i/4) and i_inner=i%4 }'
   * that's OK.
   */
  Scheduler() = default;

  /**
   * Register an Element to the scheduler.
   */
  void RegisterElement(const Element &x);

  /**
   * Finalize the registration.
   */
  void FinalizeRegistration();

  /**
   * Mark this should schedule after another.
   *
   * @param b
   * @param level
   */
  Scheduler &After(const Element &a, const Element &b, int level);
  /**
   * Mark this should schedule before another.
   * @param b
   * @param level
   */
  Scheduler &Before(const Element &a, const Element &b, int level);

  /**
   * Build and create schedule.
   */
  std::unordered_map<std::string, isl::map> BuildSchedule() const;

 private:
  /**
   * The polyhedral schedule, any schedule is performed on it.
   * We use the time-space map to record the schedule infomation, the format is borrowed from Tiramisu project:
   * [redundant,
   *
   */
  int space_size_{};
  //! Tell if the element registration is finalized.
  bool registration_finalized_{false};
  //! map from Schedule id to time schedule.
  std::unordered_map<std::string, TimeSchedule> schedule_flows_;
  //! Reversed dependency flow.
  std::unordered_map<std::string, TimeSchedule> rev_schedule_flows_;

  ScheduleGraph schedule_graph_;
};

}  // namespace poly
}  // namespace cinn
