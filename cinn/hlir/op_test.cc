#include "cinn/hlir/op.h"

#include <gtest/gtest.h>

#include <string>

#include "cinn/cinn.h"

#include "cinn/hlir/op_strategy.h"

#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

common::Shared<OpStrategy> GetStrategyTest() {
  ir::PackedFunc::body_t body = [](ir::Args args, ir::RetValue* ret) {
    Expr a = args[0];
    Expr b = args[1];
    (*ret) = Expr(pe::Add(a.as_tensor_ref(), b.as_tensor_ref(), "C").get());
  };
  ir::PackedFunc fcompute(body);
  // To do: fschedule should be an instance of pe::schedule...
  ir::PackedFunc fschedule;
  common::Shared<OpStrategy> strategy(make_shared<OpStrategy>());
  //! To build more complex strategy, we can add more than 1
  //! implementations to one Opstrategy, with different plevel.
  strategy->AddImplementation(fcompute, fschedule, "test.strategy", 10);
  return strategy;
}

TEST(Operator, GetAttr) {
  auto add           = Operator::Get("add");
  auto test_strategy = GetStrategyTest();
  Operator temp      = *add;
  temp.set_attr<common::Shared<OpStrategy>>("CINNStrategy", test_strategy);
  auto nick     = Operator::GetAttr<std::string>("nick_name");
  auto strategy = Operator::GetAttr<common::Shared<OpStrategy>>("CINNStrategy");
  ASSERT_EQ(strategy[add]->specializations[0]->implementations[0]->name, "test.strategy");
  ASSERT_EQ(add->description, "test of op Add");
  ASSERT_EQ(nick[add], "plus");
}
}  // namespace hlir
}  // namespace cinn
