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

  Placeholder<float> x("x", {M, M});

  ir::Tensor y = Compute(
      {M, M}, [=](Var i, Var j) -> Expr { return x(i, j) + 1.f; }, "y");
  LOG(INFO) << "compute: " << y->operation->as<ir::ComputeOp>()->body[0];

  ir::Tensor z = Compute(
      {M, M}, [=](Var i, Var j) -> Expr { return y(i, j) * 2.f; }, "z");

  lang::Buffer z_buffer(Float(32));
  z->Bind(z_buffer);

  LOG(INFO) << "z: " << z->operation->as<ir::ComputeOp>()->body[0];

  auto schedule = poly::CreateSchedule(z);
  LOG(INFO) << "group: " << schedule->groups.size();

  for (auto& group : schedule->groups) {
    LOG(INFO) << "group: " << group.nodes.size();
    for (auto& node : group.nodes) {
      LOG(INFO) << "node " << node->id();
    }
  }

  ASSERT_EQ(schedule->groups.size(), 2UL);
  LOG(INFO) << "Finished";
}

TEST(Call, basic) {
  Expr M(100);

  Placeholder<float> x("x", {M, Expr(10)});
  Placeholder<float> y("y", {M, Expr(10)});

  std::vector<ReturnType> return_types({{Float(32), std::vector<Expr>{{M, Expr(20)}}, "C"}});
  auto tensors = Call("lowered_fun0", {Expr(x), Expr(y)}, return_types);
}

}  // namespace lang
}  // namespace cinn
