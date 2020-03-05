#pragma once
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
#include "cinn/poly/schedule.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace poly {

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
  PolyScheduler() = default;
  explicit PolyScheduler(const std::vector<Stage *> &stages);

  /**
   * Build and create schedule.
   */
  std::map<std::string, isl::map> BuildSchedule() const;

  const std::vector<std::string> &detailed_dimension_names() const { return detailed_dimension_names_; }
};

}  // namespace poly
}  // namespace cinn
