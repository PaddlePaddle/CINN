#include "cinn/ir/intrinsic_ops.h"

namespace cinn::ir {

const char* IntrinsicOp::type_info() const { return IrNode::type_info(); }

const Type& IntrinsicOp::GetInputType(int offset) const {
  CHECK_LT(offset, input_types_.size());
  return input_types_[offset];
}
const Type& IntrinsicOp::GetOutputType(int offset) const {
  CHECK_LT(offset, output_types_.size());
  return output_types_[offset];
}

void IntrinsicOp::Verify(llvm::ArrayRef<Type> input_types, llvm::ArrayRef<Type> output_types) {
  CHECK_EQ(input_types.size(), input_types_.size());
  CHECK_EQ(output_types.size(), output_types_.size());

  for (int i = 0; i < input_types.size(); i++) {
    CHECK_EQ(input_types[i], input_types_[i]);
  }

  for (int i = 0; i < output_types.size(); i++) {
    CHECK_EQ(output_types[i], output_types_[i]);
  }
}

void IntrinsicOp::Verify(llvm::ArrayRef<Expr> inputs) {
  CHECK_EQ(inputs.size(), input_types_.size());
  for (int i = 0; i < inputs.size(); i++) {
    CHECK_EQ(inputs[i].type().IgnoreConst(), input_types_[i].IgnoreConst());
  }
}

void IntrinsicOp::Verify(llvm::ArrayRef<Expr> inputs, llvm::ArrayRef<Expr> outputs) {
  llvm::SmallVector<Type, 4> input_types, output_types;
  for (auto& e : inputs) input_types.push_back(e.type());
  for (auto& e : outputs) output_types.push_back(e.type());
  Verify(input_types, output_types);
}

Expr intrinsics::BufferGetDataHandle::Make(Expr buffer) {
  auto* n = new BufferGetDataHandle;
  n->Verify({buffer});
  n->buffer = buffer;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

Expr intrinsics::BufferGetDataConstHandle::Make(Expr buffer) {
  auto* n = new BufferGetDataConstHandle;
  n->Verify({buffer});
  n->buffer = buffer;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

Expr intrinsics::PodValueToX::Make(Expr pod_value_ptr) {
  auto* n = new PodValueToX(pod_value_ptr.type());
  n->Verify({pod_value_ptr});
  n->pod_value_ptr = pod_value_ptr;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

}  // namespace cinn::ir
