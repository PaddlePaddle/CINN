#include "cinn/frontend/paddle/cpp/block_desc.h"

namespace cinn::frontend::paddle::cpp {

template <>
VarDesc* BlockDesc::GetVar<VarDesc>(int32_t idx) {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return &vars_[idx];
}

template <>
VarDesc* BlockDesc::AddVar<VarDesc>() {
  vars_.emplace_back();
  return &vars_.back();
}

template <>
OpDesc* BlockDesc::GetOp<OpDesc>(int32_t idx) {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return &ops_[idx];
}

template <>
OpDesc* BlockDesc::AddOp<OpDesc>() {
  ops_.emplace_back();
  return &ops_.back();
}

}  // namespace cinn::frontend::paddle::cpp
