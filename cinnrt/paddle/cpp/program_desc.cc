#include "cinnrt/paddle/cpp/program_desc.h"

namespace cinnrt::paddle::cpp {
namespace framework_proto = ::paddle::framework::proto;

template <>
BlockDesc* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
BlockDesc* ProgramDesc::AddBlock<BlockDesc>() {
  blocks_.emplace_back();
  return &blocks_.back();
}

}  // namespace cinnrt::paddle::cpp
