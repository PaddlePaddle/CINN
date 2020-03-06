#pragma once

#include <glog/logging.h>
#include <isl/cpp.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/map.h"

namespace cinn {
namespace poly {

struct ComputeAtRelation;
/**
 * Stage is the basic element of polyhedral which represents a stage in CINN.
 * It supports multiple transforms such as tile, split and so on.
 */
class Stage : public Object {
 public:
  static Shared<Stage> New(const isl::set& domain, Expr expr = Expr());

  /**
   * The id of this element, should be unique across the transform.
   */
  const char* id() const;

  const Expr& expr() const { return expr_; }

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
  std::tuple<Iterator, Iterator, Iterator, Iterator>  //
  Tile(int level0, int level1, int factor0, int factor1);

  /**
   * Mark the stage compute at the level of some other stage.
   * NOTE This can only be called after all transformations are preformed, and once called, no further transform can
   * perform for that the iterators are changed, and the original `ComputeAt` level become invalid.
   */
  void ComputeAt(Stage* other, int level);

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
  const isl::map& transform() const { return transform_; }
  isl::set transformed_domain() const { return domain_.apply(transform_); }

  std::vector<ComputeAtRelation> compute_ats() const;

  //! Get the statements.
  std::vector<std::string> input_statements() const;

  virtual const char* type_info() const { return "Status"; }

  Stage() = default;

 private:
  explicit Stage(const isl::set& domain, Expr expr = Expr());

  /**
   * Initialize with an identity schedule.
   */
  void InitTransform();

 private:
  isl::set domain_;
  isl::map transform_;
  Expr expr_;
  std::map<std::string, ComputeAtRelation> compute_ats_;
};

struct ComputeAtRelation {
  Shared<Stage> stage;
  int level{-1};

  //! Check whether the stage \p self is compatible with \p stage.
  bool IsCompatible(Stage* self);
};

//! Return the corresponding inner iterator name.
inline std::string InnerName(const std::string& name);
inline std::string InnerName(const Iterator& iterator);
//! Return the corresponding inner iterator name.
inline std::string OuterName(const std::string& name);
inline std::string OuterName(const Iterator& iterator);

inline Iterator DefaultIterator(int i) { return Iterator(common::axis_name(i)); }

}  // namespace poly
}  // namespace cinn
