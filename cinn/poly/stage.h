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
#include "cinn/ir/tensor.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/map.h"

namespace cinn {
namespace poly {
using ir::DeviceAPI;

struct ComputeAtRelation;

enum class ScopeKind {
  kLocal  = 0,
  kShared = 1,
};

class StageMap;

struct StageForloopInfo {
  StageForloopInfo() = default;
  StageForloopInfo(ir::ForType for_type, ir::DeviceAPI device, uint8_t offset)
      : for_type(for_type), device(device), offset(offset) {}

  ir::ForType for_type;
  //! The offset in the \p for_type. e.g. for GPUBlock, 0 represents blockIdx.x, 1 is blockIdx.y, 2 is blockIdx.z.
  uint8_t offset;
  ir::DeviceAPI device;
};

struct ReadCacheRelation {
  //! Name of the cache tensor.
  std::string cache_name;
  //! Names of the reading tensors.
  std::vector<std::string> readers;
};

struct WriteCacheRelation {
  //! Name of the cache tensor.
  std::string cache_name;
};

//! Store the infomations about some other tensor `compute_at` this tensor.
struct ComputeAtInfo {
  ComputeAtInfo(const std::string& consumer_tensor_name,
                const std::string& producer_tensor_name,
                const std::vector<int>& adjusted_producer_shape,
                const std::vector<int>& preceding_offset_for_producer_load,
                int level)
      : consumer_tensor_name(consumer_tensor_name),
        producer_tensor_name(producer_tensor_name),
        adjusted_producer_shape(adjusted_producer_shape),
        preceding_offset_for_producer_load(preceding_offset_for_producer_load),
        level(level) {}

  std::string consumer_tensor_name;
  std::string producer_tensor_name;
  //! The shape of the buffer belong to the producer tensor after compute_at.
  //! NOTE this doesn't support dynamic dimension yet.
  std::vector<int> adjusted_producer_shape;
  //! The preceding offsets for the indice in the Loads for the producers, the offset will make the minimum indice to be
  //! 0, size of this should equal to level+1.
  std::vector<int> preceding_offset_for_producer_load;
  //! the level of the consumer tensor's transformed range.
  int level{-1};
};

/**
 * Meta infomation for tensor.
 */
struct TensorScheduleMeta {
  //! read cache relation if has one.
  std::unique_ptr<ReadCacheRelation> read_cache_relation;
  //! write cache relation if has one.
  std::unique_ptr<WriteCacheRelation> write_cache_relation;

  //! Store the information of all the other producer tensors `compute_at` this tensor.
  std::vector<ComputeAtInfo> compute_at_infos;

  bool compute_inline{false};

  //! Name of the tensors those share buffer with `this` tensor.
  std::set<std::string> tensors_to_share_buffer_with;
};

/**
 * Stage is the basic element of polyhedral which represents a stage in CINN.
 * It supports multiple transforms such as tile, split and so on.
 */
class Stage : public Object {
 public:
  static Shared<Stage> New(const isl::set& domain, Expr expr = Expr(), ir::_Tensor_* tensor = nullptr);

  TensorScheduleMeta meta;

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

  ir::_Tensor_* tensor() { return tensor_; }

  /**
   * Mark this stage to expand inplace in all the usages.
   */
  void ComputeInline();

  bool inlined() const { return meta.compute_inline; }

  /**
   * Mark this buffer should share buffer with \p other.
   */
  void ShareBufferWith(Stage* other);

  /**
   * Split the loop level of into two new loop levels.
   * @param level the level to split.
   * @param factor the extent(size) of the inner loop created after splitting.
   * @return the new outer and inner iterators.
   */
  // @{
  std::tuple<Iterator, Iterator>  //
  Split(const Iterator& level, int factor);
  std::tuple<Iterator, Iterator>  //
  Split(const std::string& level, int factor);
  std::tuple<Iterator, Iterator>  //
  Split(int level, int factor);
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

  //! Add a control dependency link to \p t.
  void CtrlDepend(const ir::Tensor& t);
  //! Get the tensors control depend on.
  const std::set<ir::Tensor>& ctrl_depends() const;

  /**
   * Create a cache Tensor and load the \p source into this buffer, replace all the reading in the readers with the
   * cache.
   * @param tensor the source memory to cache.
   * @param memory_type the memory type, "share" for CUDA share memory, "local" for CUDA local memory.
   * @param readers the readers of the \p tensor
   */
  ir::Tensor CacheRead(const std::string& memory_type, const std::vector<ir::Tensor>& readers, poly::StageMap stages);

  /**
   * Create a cache for write to the original tensor.
   * @param tensor the tensor to create the cache for.
   * @param memory_type "share" for CUDA share memory, "local" for CUDA local memory.
   */
  ir::Tensor CacheWrite(const std::string& memory_type, poly::StageMap stages);

  /**
   * Set thread scope.
   */
  void SetScope(ScopeKind scope) { scope_ = scope; }

  /**
   * Get thread scope.
   */
  ScopeKind scope() const { return scope_; }

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

  /** Get the final level after all the transforms.
   * The level will be affected by some schedule like ComputeAt, this will return the right level.
   *
   * @param level the level in schedule.
   */
  int GetTransformedLevel(int level);

  //! Get the statements.
  std::vector<std::string> input_statements() const;

  virtual const char* type_info() const { return __type_info__; }

  inline const ir::VectorizeInfo& vectorize_info() const { return vectorize_info_; }
  inline const std::set<int>& unroll_info() const { return unroll_info_; }

  /*
  const std::set<std::string>& extra_depend_stages() const { return extra_depend_stages_; }
  void set_extra_depend_stages(const std::set<std::string>& x) { extra_depend_stages_ = x; }
  void add_extra_depend_stage(const std::string& statement) { extra_depend_stages_.insert(statement); }
   */

  const std::map<int /*level*/, StageForloopInfo>& forloop_infos() const { return forloop_infos_; }

  bool has_expression() const;

  Stage() = default;

  void ComputeAtSchedule(Stage* other, int level, ComputeAtKind kind = kComputeAtAuto);

  ir::Tensor LookupCtrlDepend(const std::string& tensor_name) const;

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

  static constexpr char* __type_info__ = "Stage";

 private:
  isl::set domain_;
  isl::map transform_;
  Expr expr_;
  // this compute_at some other stages.
  std::map<std::string, ComputeAtRelation> compute_ats_;
  ir::VectorizeInfo vectorize_info_;
  //! The for-loop levels to unroll.
  std::set<int> unroll_info_;
  //! Record some forloop levels' information.
  std::map<int /*level*/, StageForloopInfo> forloop_infos_;
  //! A weak reference to the tensor.
  ir::_Tensor_* tensor_{};
  //! Thread scope.
  ScopeKind scope_{ScopeKind::kLocal};

  std::set<ir::Tensor> ctrl_depends_;

  std::set<int> locked_axis_;

  friend isl_map* __isl_give GatherAccesses(Stage* stage, const std::string& tensor_name);
  friend class PolyGroupScheduler;
};

std::vector<std::pair<std::string, std::string>> ExtractExtraDepLinksFromStages(const std::vector<Stage*>& stages);

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

class _StageMap_ : public Object {
 public:
  /**
   * Get a stage from the stage map.
   * NOTE The stage should exists, or it will abort.
   */
  // @{
  Stage* operator[](const ir::Tensor& tensor);
  const Stage* operator[](const ir::Tensor& tensor) const;
  Stage* operator[](const ir::_Tensor_* tensor);
  const Stage* operator[](const ir::_Tensor_* tensor) const;
  // @}

  //! Insert a stage into the map, it will replace if an older one exists.
  Stage* Insert(const ir::Tensor& key, Stage* stage);
  //! Insert a stage only if not exists.
  Stage* InsertLazily(const ir::Tensor& key);

  //! Lookup a tensor from the map, return nullptr if not exists.
  Stage* Lookup(const std::string& name) const;

  inline size_t size() const { return data_.size(); }

  const char* type_info() const override { return __type_info__; }

  static constexpr const char* __type_info__ = "StageMap";

 private:
  std::unordered_map<std::string, Shared<Stage>> data_;

  friend class StageMap;
};

class StageMap : public Shared<_StageMap_> {
 public:
  StageMap() : Shared(new _StageMap_) {}

  Stage* operator[](const ir::Tensor& tensor) { return (*self())[tensor]; }
  const Stage* operator[](const ir::Tensor& tensor) const { return (*self())[tensor]; }
  Stage* operator[](const ir::_Tensor_* tensor) { return (*self())[tensor]; }
  const Stage* operator[](const ir::_Tensor_* tensor) const { return (*self())[tensor]; }

  auto begin() const { return self()->data_.begin(); }
  auto end() const { return self()->data_.end(); }
};

StageMap CreateStages(const std::vector<ir::Tensor>& tensors);

}  // namespace poly
}  // namespace cinn
