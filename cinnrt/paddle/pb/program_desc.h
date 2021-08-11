#pragma once
#include <glog/logging.h>

#include <string>
#include <vector>

#include "cinnrt/paddle/cpp/desc_api.h"
#include "cinnrt/paddle/framework.pb.h"

namespace cinnrt::paddle::pb {
namespace framework_proto = ::paddle::framework::proto;

class ProgramDesc : public cpp::ProgramDescAPI {
 public:
  ProgramDesc() = delete;

  explicit ProgramDesc(framework_proto::ProgramDesc *desc) : desc_(desc) { CHECK(desc_); }

  framework_proto::ProgramDesc *Proto() { return desc_; }

  const framework_proto::ProgramDesc &ReadonlyProto() const { return *desc_; }

  size_t BlocksSize() const override { return desc_->blocks_size(); }

  void ClearBlocks() override { desc_->clear_blocks(); }

  template <typename T>
  T *GetBlock(int32_t idx);

  template <typename T>
  T *AddBlock();

  bool HasVersion() const override { return desc_->has_version(); }

  int64_t Version() const override { return desc_->version().version(); }

  void SetVersion(int64_t version) override { desc_->mutable_version()->set_version(version); }

 private:
  framework_proto::ProgramDesc *desc_;  // not_own
};

}  // namespace cinnrt::paddle::pb
