#pragma once
#include "cinnrt/paddle/cpp/desc_api.h"
#include "cinnrt/paddle/cpp/op_desc.h"
#include "cinnrt/paddle/cpp/var_desc.h"

namespace cinnrt::paddle::cpp {

/*
 * The cpp::BlockDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::BlockDesc.
 */
class BlockDesc : public BlockDescAPI {
 public:
  BlockDesc() = default;

  int32_t Idx() const override { return idx_; }

  void SetIdx(int32_t idx) override { idx_ = idx; }

  int32_t ParentIdx() const override { return parent_idx_; }

  void SetParentIdx(int32_t idx) override { parent_idx_ = idx; }

  size_t VarsSize() const override { return vars_.size(); }

  void ClearVars() override { vars_.clear(); }

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T* AddVar();

  size_t OpsSize() const override { return ops_.size(); }

  void ClearOps() override { ops_.clear(); }

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T* AddOp();

  int32_t ForwardBlockIdx() const override { return forward_block_idx_; }

  void SetForwardBlockIdx(int32_t idx) override { forward_block_idx_ = idx; }

 private:
  int32_t idx_;
  int32_t parent_idx_;
  std::vector<OpDesc> ops_;
  std::vector<VarDesc> vars_;
  int32_t forward_block_idx_;
};

}  // namespace cinnrt::paddle::cpp
