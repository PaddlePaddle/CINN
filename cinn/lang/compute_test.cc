#include "cinn/lang/compute.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace lang {

TEST(Compute, basic) {
  Expr M(100);
  Expr N(100);

  Placeholder<float> x("x", {M, N});

  ir::Tensor y = Compute(
      {100, 100}, [=](Var i, Var j) -> Expr { return x(i, j) + 1.f; }, "y");
  LOG(INFO) << "compute: " << y->operaion->As<ir::ComputeOp>()->body[0];

  ir::Tensor z = Compute(
      {100, 100}, [=](Var i, Var j) -> Expr { return y(i, j) * 2.f; }, "z");

  lang::Buffer z_buffer(Float(32));
  z->Bind(z_buffer);

  LOG(INFO) << "z: " << z->operaion->As<ir::ComputeOp>()->body[0];

  auto schedule = poly::CreateSchedule(z);
  LOG(INFO) << "group: " << schedule->gened_groups().size();

  for (auto& group : schedule->gened_groups()) {
    LOG(INFO) << "group: " << group.nodes.size();
    for (auto& node : group.nodes) {
      LOG(INFO) << "node " << node->id();
    }
  }

  ASSERT_EQ(schedule->gened_groups().size(), 1UL);
  EXPECT_EQ(schedule->gened_groups().front().nodes[0]->id(), "z");
  LOG(INFO) << "Finished";
}

}  // namespace lang
}  // namespace cinn
