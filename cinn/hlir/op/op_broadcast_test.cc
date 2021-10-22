// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
namespace framework {

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

TEST(Operator, Operator_ElementWise_Add_Test0) {
  auto add      = Operator::Get("elementwise_add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attrs;
  std::vector<ir::Tensor> inputs{A.tensor(), B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[add](attrs, inputs, type, {{M.as_int32(), N.as_int32()}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A), common::CINNValue(B)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  ASSERT_EQ(rets.size(), 2UL);
  rets = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  Module::Builder builder("module0", target);
  auto func = Lower("add1", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;
  builder.AddFunction(func);
  LOG(INFO) << "func:\n" << func;

  ASSERT_EQ(impl->name, "strategy.elementwise_add.x86");
  ASSERT_EQ(add->description, "elementwise_add function");

  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();
  jit->Link(module);
  auto fn = jit->Lookup("add1");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);
  cinn_buffer_t *A_buf;
  cinn_buffer_t *B_buf;
  int set_value = 0;
  if (set_value != 0) {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_align(512).set_val(set_value).Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_align(512).set_val(set_value).Build();
  } else {
    A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_align(512).set_random().Build();
    B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_align(512).set_random().Build();
  }
  auto *C_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_align(512).set_zero().Build();

  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  auto *ad = reinterpret_cast<float *>(A_buf->memory);
  auto *bd = reinterpret_cast<float *>(B_buf->memory);
  auto *cd = reinterpret_cast<float *>(C_buf->memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    ASSERT_NEAR(cd[i], ad[i] + bd[i], 1e-5);
  }
}

TEST(Operator, Operator_ElementWise_Add_Test1) {
  auto add      = Operator::Get("elementwise_add");
  Operator temp = *add;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {N});

  NodeAttr attrs;
  attrs.attr_store["axis"] = 1;
  std::vector<ir::Tensor> inputs{A.tensor(), B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target            = common::DefaultHostTarget();
  auto impl                        = OpStrategy::SelectImpl(strategy[add](attrs, inputs, type, {{100, 32}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A), common::CINNValue(B)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  ASSERT_EQ(rets.size(), 2UL);
  rets = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("add1", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;
  std::cout << func;

  ASSERT_EQ(impl->name, "strategy.elementwise_add.x86");
  ASSERT_EQ(add->description, "elementwise_add function");
}

TEST(Operator, Operator_BroadcastTo) {
  auto broadcast_to      = Operator::Get("broadcast_to");
  Operator temp = *broadcast_to;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1);
  Placeholder<float> B("B", {N});

  NodeAttr attrs;
  std::vector<int> out_shape = {16};
  attrs.attr_store["out_shape"] = out_shape;

  std::vector<int> broadcast_axes = {0};
  attrs.attr_store["broadcast_axes"] = broadcast_axes;

  std::vector<ir::Tensor> inputs{B.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target            = common::DefaultHostTarget();

  auto impl                        = OpStrategy::SelectImpl(strategy[broadcast_to](attrs, inputs, type, {out_shape}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(B)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);

  ASSERT_EQ(rets.size(), 2UL);
  rets = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }

  auto func = Lower("broadcast_to", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;
  std::cout << func;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
