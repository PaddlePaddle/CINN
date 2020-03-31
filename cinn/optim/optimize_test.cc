#include "cinn/optim/optimize.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {

TEST(Optimize, Unroll) {
  Placeholder<float> A("A", {100, 20});
  lang::Buffer buf(Float(32));

  auto C = Compute({100, 20}, [&](Var i, Var j) { return A(i, j) + 1.f; });
  C->Bind(buf);

  C->stage()->Split(1, 5);
  C->stage()->Unroll(2);

  auto func = Lower("matmul", {A, C});

  auto out = R"ROC(
{
  for (i, 100)
  {
    for (j_outer, 4)
    {
      {
        tensor_2[i, (5 * j_outer)] = (1 + A[i, (5 * j_outer)])
      }
      {
        tensor_2[i, (1 + (5 * j_outer))] = (1 + A[i, (1 + (5 * j_outer))])
      }
      {
        tensor_2[i, (2 + (5 * j_outer))] = (1 + A[i, (2 + (5 * j_outer))])
      }
      {
        tensor_2[i, (3 + (5 * j_outer))] = (1 + A[i, (3 + (5 * j_outer))])
      }
      {
        tensor_2[i, (4 + (5 * j_outer))] = (1 + A[i, (4 + (5 * j_outer))])
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(utils::GetStreamCnt(func->body)), utils::Trim(out));
}

}  // namespace optim
}  // namespace cinn
