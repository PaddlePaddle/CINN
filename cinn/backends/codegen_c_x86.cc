#include "cinn/backends/codegen_c_x86.h"

namespace cinn {
namespace backends {

void CodeGenCX86::Visit(const ir::Add *op) {
  auto a = op->a;
  auto b = op->b;
  CHECK_EQ(a.type(), b.type());

  // scalar.
  if (a.type().lanes() == 1) {
    CodeGenC::Visit(op);
    return;
  }

  // TODO(Superjomn) Consider support BLAS.
  CHECK(SupportsAVX256() || SupportsAVX512());
  int bits = a.type().bits() * a.type().lanes();

  // prepare argument.

  if (SupportsAVX512() && bits >= 512) {
    CHECK_EQ(bits % 512, 0) << "the bits of computation should be times of 512";
    int times = bits / 512;
  }
}

void CodeGenCX86::Visit(const ir::Load *op) {
  Expr dense_strided_ramp = detail::StridedRampBase(op->index, 1);
  if (dense_strided_ramp.defined()) {  // Loading a continuous Ramp address.
    CHECK(op->type().is_vector());

    int bits = op->type().bits() * op->type().lanes();
    if (SupportsAVX512()) {
      CHECK_EQ(bits, 512);
      os() << "cinn_avx512_load(" << op->tensor.As<ir::_Tensor_>()->name << ")";
    } else if (SupportsAVX256()) {
      CHECK_EQ(bits, 256);
      os() << "cinn_avx256_load(" << op->tensor.As<ir::_Tensor_>()->name << ")";
    } else {
      CodeGenC::Visit(op);
    }
  } else {
    CodeGenC::Visit(op);
  }
}

}  // namespace backends
}  // namespace cinn
