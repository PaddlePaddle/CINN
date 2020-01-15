#include "cinn/ir/ir_printer.h"
#include <gtest/gtest.h>
#include <sstream>

namespace cinn {
namespace ir {

TEST(ir_printer, test) {
  std::stringstream ss;
  IrPrinter printer(ss);

  Builder builder;
  Expr one(1);
  Expr two(2);
  auto add = builder.MakeExpr<Add>(one, two);

  printer.Print(add);

  LOG(INFO) << ss.str();
}

}  // namespace ir
}  // namespace cinn
