#include "cinn/frontend/paddle/pb/block_desc.h"

namespace cinn::frontend::paddle::pb {

template <>
framework_proto::VarDesc* BlockDesc::GetVar<framework_proto::VarDesc>(int32_t idx) {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return desc_->mutable_vars(idx);
}

template <>
framework_proto::VarDesc* BlockDesc::AddVar<framework_proto::VarDesc>() {
  return desc_->add_vars();
}

template <>
framework_proto::OpDesc* BlockDesc::GetOp<framework_proto::OpDesc>(int32_t idx) {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return desc_->mutable_ops(idx);
}

template <>
framework_proto::OpDesc* BlockDesc::AddOp<framework_proto::OpDesc>() {
  return desc_->add_ops();
}

}  // namespace cinn::frontend::paddle::pb
