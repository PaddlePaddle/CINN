#include "cinn/optim/ir_copy.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

TEST(IrCopy, basic) {
  Expr a(1.f);
  auto aa = IRCopy(a);
  LOG(INFO) << "aa " << aa;
}

}  // namespace optim
}  // namespace cinn
