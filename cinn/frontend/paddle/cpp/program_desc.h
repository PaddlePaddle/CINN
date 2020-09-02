#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "cinn/frontend/paddle/cpp/block_desc.h"
#include "cinn/frontend/paddle/cpp/desc_api.h"

namespace cinn::frontend::paddle::cpp {

/*
 * The cpp::ProgramDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::ProgramDesc.
 */
class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;

  size_t BlocksSize() const override { return blocks_.size(); }

  void ClearBlocks() override { blocks_.clear(); }

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T* AddBlock();

  // Just return default versoin
  // TODO(sangoly): refine this
  bool HasVersion() const override { return true; }

  int64_t Version() const override { return version_; }

  void SetVersion(int64_t version) override { version_ = version; }

 private:
  int64_t version_;
  std::vector<cpp::BlockDesc> blocks_;
};

}  // namespace cinn::frontend::paddle::cpp
