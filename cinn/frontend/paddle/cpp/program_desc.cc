#include "cinn/frontend/paddle/cpp/program_desc.h"

namespace cinn::frontend::paddle::cpp {

template <>
BlockDesc* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
const BlockDesc& ProgramDesc::GetConstBlock<BlockDesc>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return blocks_[idx];
}

template <>
BlockDesc* ProgramDesc::AddBlock<BlockDesc>() {
  blocks_.emplace_back();
  return &blocks_.back();
}

}  // namespace cinn::frontend::paddle::cpp
