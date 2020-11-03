#include "cinn/ir/intrinsic_ops.h"
#include <gtest/gtest.h>

namespace cinn::ir {

TEST(IntrinsicOp, basic) {
  Expr buffer(1);
  buffer->set_type(type_of<cinn_buffer_t*>());
  auto op   = intrinsics::BufferGetDataHandle::Make(buffer);
  auto* ptr = op.As<IntrinsicOp>();
  ASSERT_TRUE(ptr);
  auto* obj = llvm::dyn_cast<intrinsics::BufferGetDataHandle>(ptr);
  ASSERT_TRUE(obj);
}

}  // namespace cinn::ir
