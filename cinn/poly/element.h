#pragma once

#include <glog/logging.h>
#include <isl/cpp.h>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "cinn/poly/domain.h"
#include "cinn/poly/map.h"

namespace cinn {
namespace poly {

/**
 * Element is the basic element of polyhedral which represents a stage in CINN.
 * It supports multiple transforms such as tile, split and so on.
 */
class Element {
 public:
  explicit Element(const isl::set& domain);

  /**
   * The id of this element, should be unique across the schedule.
   */
  const char* id() const;

  /**
   * Split the loop level of into two new loop levels.
   * @param level the level to split.
   * @param factor the extent(size) of the inner loop created after splitting.
   * @return the new outer and inner iterators.
   */
  std::tuple<Iterator, Iterator>  //
      Split(const Iterator& level, int factor);
  std::tuple<Iterator, Iterator>  //
      Split(const std::string& level, int factor);

  /**
   * Reorder the iterators.
   * @param order the order of all the iterators.
   */
  void Reorder(const std::vector<Iterator>& order);

  /**
   * Tile the two loop levels \p level0 and \p level1 with rectangular tiling.
   * @param level0 the first level.
   * @param level1 the second level.
   * @param factor0 tiling size of the first level.
   * @param factor1 tiling size of the second level.
   * @return the new iterators.
   */
  std::tuple<Iterator, Iterator, Iterator, Iterator>  //
      Tile(const Iterator& level0, const Iterator& level1, int factor0, int factor1);

  /**
   * Apply loop skewing on the loop levels \p i and \p j with a skewing factor of \p factor.
   */
  std::tuple<Iterator, Iterator>  //
      Skew(const Iterator& i, const Iterator& j, int factor);

  /**
   * Fuse two levels and return the new level.
   * @param level0
   * @param level1
   * @return
   */
  Iterator Fuse(const Iterator& level0, const Iterator& level1);

  const isl::set& domain() const { return domain_; }
  const isl::map& schedule() const { return schedule_; }

 private:
  /**
   * Initialize with an identity schedule.
   */
  void InitSchedule();

 private:
  isl::set domain_;
  isl::map schedule_;
};

//! Return the corresponding inner iterator name.
inline std::string InnerName(const std::string& name);
inline std::string InnerName(const Iterator& iterator);
//! Return the corresponding inner iterator name.
inline std::string OuterName(const std::string& name);
inline std::string OuterName(const Iterator& iterator);

}  // namespace poly
}  // namespace cinn
