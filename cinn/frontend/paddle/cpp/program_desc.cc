#include "cinn/frontend/paddle/cpp/program_desc.h"

namespace cinn::frontend::paddle::cpp {
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

}  // namespace cinn::frontend::paddle::cpp
