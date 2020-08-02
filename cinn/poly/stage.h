#pragma once

#include <glog/logging.h>
#include <isl/cpp.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/map.h"

namespace cinn {
namespace ir {
class _Tensor_;
class Tensor;
}  // namespace ir

namespace poly {
using ir::DeviceAPI;

struct ComputeAtRelation;

//! The strategy to deal with the rest domain of a split.
enum class SplitRestStrategy {
  //! Leave it unchanged.
  kAuto,
  //! Separate the rest.
  kSeparate,
};

struct StageForloopInfo {
  StageForloopInfo() = default;
  StageForloopInfo(ir::ForType for_type, ir::DeviceAPI device, uint8_t offset)
      : for_type(for_type), device(device), offset(offset) {}

  ir::ForType for_type;
  //! The offset in the \p for_type. e.g. for GPUBlock, 0 represents blockIdx.x, 1 is blockIdx.y, 2 is blockIdx.z.
  uint8_t offset;
  ir::DeviceAPI device;
};

/**
 * Stage is the basic element of polyhedral which represents a stage in CINN.
 * It supports multiple transforms such as tile, split and so on.
 */
class Stage : public Object {
 public:
  static Shared<Stage> New(const isl::set& domain, Expr expr = Expr(), ir::_Tensor_* tensor = nullptr);

  /**
   * The id of this element, should be unique across the transform.
   */
  const char* id() const;

  //! Expression contained in this stage.
  const Expr& expr() const { return expr_; }

  //! Get the i-th axis.
  Iterator axis(int i) const;
  //! Get the axis named \p i.
  Iterator axis(const std::string& i) const;

  std::vector<std::string> axis_names() const;

  /**
   * Mark this stage to expand inplace in all the usages.
   */
  void ComputeInline();

  /**
   * Mark this buffer should share buffer with \p other.
   */
  void ShareBufferWith(ir::Tensor other);

  /**
   * Split the loop level of into two new loop levels.
   * @param level the level to split.
   * @param factor the extent(size) of the inner loop created after splitting.
   * @return the new outer and inner iterators.
   */
  // @{
  std::tuple<Iterator, Iterator>  //
  Split(const Iterator& level, int factor, SplitRestStrategy strategy = SplitRestStrategy::kAuto);
  std::tuple<Iterator, Iterator>  //
  Split(const std::string& level, int factor, SplitRestStrategy strategy = SplitRestStrategy::kAuto);
  std::tuple<Iterator, Iterator>  //
  Split(int level, int factor, SplitRestStrategy strategy = SplitRestStrategy::kAuto);
  // @}

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
   * Vectorize the stage in \p level.
   * @param level
   */
  void Vectorize(int level, int factor);
  void Vectorize(const std::string& axis, int factor);
  void Vectorize(const Iterator& axis, int factor);

  /**
   * Unroll a for-loop.
   */
  void Unroll(int level);
  void Unroll(const std::string& level);
  void Unroll(const Iterator& level);

  void Bind(int level, const std::string& axis);

  enum ComputeAtKind {
    kComputeAtAuto,
    kComputeAtBefore,
    kComputeAtAfter,
  };

  /**
   * \brief Mark the stage compute at the level of some other stage.
   *
   * NOTE This can only be called after all transformations are preformed, and once called, no further transform can
   * perform for that if the iterators are changed, the original `ComputeAt` level will become invalid.
   *
   * @param other the target stage to compute at.
   * @param level the level of \p other's forloop to compute at
   * @param kind the position compared to other, can be Before, After or Unknown.
   */
  void ComputeAt(Stage* other,
                 int level,
                 ComputeAtKind kind                    = kComputeAtAuto,
                 const std::string& cached_tensor_name = "");

  /**
   * Apply loop skewing on the loop levels \p i and \p j with a skewing factor of \p factor.
   * TODO(Superjomn) Refine this transform.
   */
  std::tuple<Iterator, Iterator>  //
  Skew(const Iterator& i, const Iterator& j, int factor);

  //! Set GPU thread axis.
  // @{
  void GpuThreads(const std::vector<int>& levels, DeviceAPI device = DeviceAPI::GPU);
  void GpuThreads(const Iterator& thread_x, DeviceAPI device = DeviceAPI::GPU);
  void GpuThreads(const Iterator& thread_x, const Iterator& thread_y, DeviceAPI device = DeviceAPI::GPU);
  void GpuThreads(const Iterator& thread_x,
                  const Iterator& thread_y,
                  const Iterator& thread_z,
                  DeviceAPI device = DeviceAPI::GPU);
  void GpuThreads(const std::vector<Iterator>& iters, DeviceAPI device);
  // @}

  //! Set GPU block axis.
  // @{
  void GpuBlocks(const std::vector<int>& levels, DeviceAPI device = DeviceAPI::GPU);
  void GpuBlocks(const Iterator& block_x, DeviceAPI device = DeviceAPI::GPU);
  void GpuBlocks(const Iterator& block_x, const Iterator& block_y, DeviceAPI device = DeviceAPI::GPU);
  void GpuBlocks(const Iterator& block_x,
                 const Iterator& block_y,
                 const Iterator& block_z,
                 DeviceAPI device = DeviceAPI::GPU);
  void GpuBlocks(const std::vector<Iterator>& iters, DeviceAPI device);
  // @}

  // Add a control dependency link to \p t.
  void CtrlDepend(const ir::Tensor& t);

  /**
   * Create a cache Tensor and load the \p source into this buffer, replace all the reading in the readers with the
   * cache.
   * @param tensor the source memory to cache.
   * @param memory_type the memory type, "share" for CUDA share memory, "local" for CUDA local memory.
   * @param readers the readers of the \p tensor
   */
  ir::Tensor CacheRead(const std::string& memory_type, const std::vector<ir::Tensor>& readers);

  /**
   * Create a cache for write to the original tensor.
   * @param tensor the tensor to create the cache for.
   * @param memory_type "share" for CUDA share memory, "local" for CUDA local memory.
   */
  ir::Tensor CacheWrite(const std::string& memory_type);

  /**
   * \brief Fuse two forloop levels and return the new level.
   * @param level0 the first level.
   * @param level1 the second level.
   * @return the new level.
   */
  Iterator Fuse(const Iterator& level0, const Iterator& level1);
  Iterator Fuse(int level0, int level1);
  Iterator Fuse(const std::string& level0, const std::string& level1);

  const isl::set& domain() const { return domain_; }
  const isl::map& transform() const { return transform_; }
  isl::set transformed_domain() const;

  // Dealing with the `ComputateAt` transform.
  std::vector<ComputeAtRelation> compute_ats() const;

  //! Get the level-th dimensional name.
  std::string ith_dim_name(int level);
  //! Get the i-th iterator.
  Iterator ith_iterator(int level);

  //! Get the statements.
  std::vector<std::string> input_statements() const;

  virtual const char* type_info() const { return "Status"; }

  inline const ir::VectorizeInfo& vectorize_info() const { return vectorize_info_; }
  inline const std::set<int>& unroll_info() const { return unroll_info_; }

  const std::set<std::string>& extra_depend_stages() const { return extra_depend_stages_; }
  void set_extra_depend_stages(const std::set<std::string>& x) { extra_depend_stages_ = x; }
  void add_extra_depend_stage(const std::string& statement) { extra_depend_stages_.insert(statement); }

  const std::map<int /*level*/, StageForloopInfo>& forloop_infos() const { return forloop_infos_; }

  bool has_expression() const;

  Stage() = default;

  void ComputeAtSchedule(Stage* other, int level, ComputeAtKind kind = kComputeAtAuto);

 private:
  explicit Stage(const isl::set& domain, Expr expr = Expr(), ir::_Tensor_* tensor = nullptr);

  /**
   * Initialize with an identity schedule.
   */
  void InitTransform();

  void AddForloopInfo(int level, const StageForloopInfo& info);

  //! Lock the \p level-th axis and disallow the futher schedules on this axis.
  void LockAxis(uint32_t level);
  //! Unlock the \p level-th axis.
  void UnlockAxis(uint32_t level);
  //! Tell if the \p level -th axis is locked.
  bool is_axis_locked(uint32_t level) const;
  //! Assert that the axis is not locked, abort if fail.
  void AssertAxisIsNotLocked(uint32_t level);

  //! Get number of transform output dimensions, this equals to the number of forloops in generated code.
  inline int n_in_dims() const { return isl_map_dim(transform_.get(), isl_dim_in); }
  //! Get number of transform output dimensions, this equals to the number of dimensions of corresponding tensor.
  inline int n_out_dims() const { return isl_map_dim(transform_.get(), isl_dim_out); }

 private:
  isl::set domain_;
  isl::map transform_;
  Expr expr_;
  // this compute_at some other stages.
  std::map<std::string, ComputeAtRelation> compute_ats_;
  ir::VectorizeInfo vectorize_info_;
  //! The for-loop levels to unroll.
  std::set<int> unroll_info_;
  // TODO(Superjomn) Remove this.
  std::map<std::string /*iterator name*/, SplitRestStrategy> split_strageties_;
  //! The other stages it depends.
  std::set<std::string> extra_depend_stages_;
  //! Record some forloop levels' information.
  std::map<int /*level*/, StageForloopInfo> forloop_infos_;
  //! A weak reference to the tensor.
  ir::_Tensor_* tensor_{};

  std::set<int> locked_axis_;

  friend isl_map* __isl_give GatherAccesses(Stage* stage, const std::string& tensor_name);
  friend class PolyGroupScheduler;
};

std::vector<std::pair<std::string, std::string>> ExtractExtraDepLinksFromStages(const std::vector<Stage*>& stages);
std::vector<std::pair<std::string, std::string>> ExtractLinksFromCalls(const std::vector<ir::Tensor>& tensors,
                                                                       bool with_placeholder = false);

//! This stage compute_at some other stage.
struct ComputeAtRelation {
  //! the other stage.
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

/**
 * Collect the access to a tensor named \p tensor_name in \p stage.
 */
std::vector<isl::map> GatherAccesses(const Stage* stage, const std::string& tensor_name);

}  // namespace poly
}  // namespace cinn
