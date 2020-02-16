#include "cinn/optim/remove_nested_block.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

TEST(RemoveNestedBlock, basic) {
  auto block0 = ir::Block::Make({Expr(1.f), Expr(1.f)});
  auto block1 = ir::Block::Make({block0});
  auto e      = Expr(block1);

  std::string origin = utils::GetStreamCnt(e);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
{
  {
    1
    1
  }
}
  )ROC"));

  std::cout << "origin:\n" << e << std::endl;

  RemoveNestedBlock(&e);

  std::cout << "e:\n" << e << std::endl;

  EXPECT_EQ(utils::GetStreamCnt(e), utils::Trim(R"ROC(
{
  1
  1
}
  )ROC"));
}

}  // namespace optim
}  // namespace cinn
