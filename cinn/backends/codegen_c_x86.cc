#include "cinn/backends/codegen_c_x86.h"

namespace cinn {
namespace backends {

void CodeGenCX86::Visit(const ir::Add *op) { VisitBinaryOp(op, op->a, op->b, "add"); }
void CodeGenCX86::Visit(const ir::Sub *op) { VisitBinaryOp(op, op->a, op->b, "sub"); }
void CodeGenCX86::Visit(const ir::Mul *op) { VisitBinaryOp(op, op->a, op->b, "mul"); }
void CodeGenCX86::Visit(const ir::Div *op) { VisitBinaryOp(op, op->a, op->b, "div"); }

void CodeGenCX86::Visit(const ir::Load *op) {
  Expr dense_strided_ramp = detail::StridedRampBase(op->index, 1);
  if (dense_strided_ramp.defined()) {  // Loading a continuous Ramp address.
    CHECK(op->type().is_vector());

    int bits = op->type().bits() * op->type().lanes();
    if (SupportsAVX512()) {
      CHECK_EQ(bits, 512);
      os() << "cinn_avx512_load(";
      PrintAbsAddr(op);
      os() << ")";
    } else if (SupportsAVX256()) {
      CHECK_EQ(bits, 256);
      os() << "cinn_avx256_load(";
      PrintAbsAddr(op);
      os() << ")";
    } else {
      CodeGenC::Visit(op);
    }
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenCX86::Visit(const ir::Store *op) {
  if (op->type().lanes() == 1) {
    LOG(INFO) << "store lanes 1 " << op->index;
    CodeGenC::Visit(op);
    return;
  }

  int bits = op->type().bits() * op->type().lanes();
  if (SupportsAVX512()) {
    CHECK_EQ(bits, 512);
    os() << "cinn_avx512_store(" << op->tensor.As<ir::_Tensor_>()->name << ", " << op->value << ")";
  } else if (SupportsAVX256()) {
    CHECK_EQ(bits, 256);
    os() << "cinn_avx256_store(" << op->tensor.As<ir::_Tensor_>()->name << ", " << op->value << ")";
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenCX86::PrintAbsAddr(const ir::Load *op) {
  os() << op->tensor.As<ir::_Tensor_>()->name << " + ";

  auto *ramp_n = op->index.As<ir::Ramp>();
  if (ramp_n) {
    CHECK(!ramp_n->base.As<ir::Ramp>()) << "base of a Ramp node should not be Ramp type";
    Print(ramp_n->base);
  } else {
    Print(op->index);
  }
}

void CodeGenCX86::PrintVecInputArgument(const Expr *op) {
  int bits          = op->type().bits() * op->type().lanes();
  auto *broadcast_n = op->As<ir::Broadcast>();

  if (op->type().lanes() == 1 || broadcast_n) {
    Expr value = op->type().lanes() == 1 ? *op : broadcast_n->value;

    if (SupportsAVX512()) {
      os() << "cinn_avx512_set1(";
      Print(value);
      os() << ")";
    } else if (SupportsAVX256()) {
      os() << "cinn_avx256_set1(";
      Print(value);
      os() << ")";
    } else {
      NOT_IMPLEMENTED
    }
  } else {
    auto *load_n = op->As<ir::Load>();

    if (load_n) {
      Visit(load_n);
      return;
    }

    NOT_IMPLEMENTED
  }
}

}  // namespace backends
}  // namespace cinn
