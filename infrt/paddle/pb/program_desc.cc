#include "infrt/paddle/pb/program_desc.h"

#include <algorithm>
#include <limits>

namespace infrt::paddle::pb {

template <>
framework_proto::BlockDesc* ProgramDesc::GetBlock<framework_proto::BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return desc_->mutable_blocks(idx);
}

template <>
framework_proto::BlockDesc* ProgramDesc::AddBlock<framework_proto::BlockDesc>() {
  return desc_->add_blocks();
}

}  // namespace infrt::paddle::pb
