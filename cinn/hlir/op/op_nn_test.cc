#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/nn.h"

namespace cinn {
namespace hlir {
namespace framework {

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

TEST(Operator, Operator_Pool2d_Test0) {
  auto pool2d   = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), C(3), H(8), W(8);
  Placeholder<float> A("A", {N, C, H, W});

  NodeAttr attrs;
  std::vector<int> kernel_size     = {2, 2};
  std::vector<int> stride_size     = {2, 2};
  std::vector<int> padding_size    = {1, 1, 1, 1};
  std::string pool_type            = "max";
  attrs.attr_store["kernel_size"]  = kernel_size;
  attrs.attr_store["stride_size"]  = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"]    = pool_type;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](attrs, inputs, type, {{1, 3, 10, 10}, {1, 3, 5, 5}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("pool2d", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("pool2d");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {1, 3, 8, 8}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {1, 3, 10, 10}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {1, 3, 5, 5}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(pool2d->description, "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool2d_Test1) {
  auto pool2d   = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), C(3), H(8), W(8);
  Placeholder<float> A("A", {N, C, H, W});

  NodeAttr attrs;
  std::vector<int> kernel_size     = {2, 2};
  std::vector<int> stride_size     = {2, 2};
  std::vector<int> padding_size    = {1, 1, 1, 1};
  std::string pool_type            = "avg";
  attrs.attr_store["kernel_size"]  = kernel_size;
  attrs.attr_store["stride_size"]  = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"]    = pool_type;
  attrs.attr_store["ceil_mode"]    = true;
  attrs.attr_store["exclusive"]    = false;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](attrs, inputs, type, {{1, 3, 11, 11}, {1, 3, 5, 5}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("pool2d", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("pool2d");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {1, 3, 8, 8}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {1, 3, 11, 11}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {1, 3, 5, 5}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(pool2d->description, "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool2d_Test2) {
  auto pool2d   = Operator::Get("pool2d");
  Operator temp = *pool2d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), H(8), W(8), C(3);
  Placeholder<float> A("A", {N, H, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size     = {2, 2};
  std::vector<int> stride_size     = {2, 2};
  std::vector<int> padding_size    = {1, 1, 1, 1};
  std::string pool_type            = "avg";
  std::string data_format          = "NHWC";
  attrs.attr_store["kernel_size"]  = kernel_size;
  attrs.attr_store["stride_size"]  = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"]    = pool_type;
  attrs.attr_store["ceil_mode"]    = true;
  attrs.attr_store["exclusive"]    = true;
  attrs.attr_store["data_format"]  = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool2d](attrs, inputs, type, {{1, 11, 11, 3}, {1, 5, 5, 3}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("pool2d", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("pool2d");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {1, 8, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {1, 11, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {1, 5, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool2d.x86");
  ASSERT_EQ(pool2d->description, "Do pooling on the height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool3d_Test0) {
  auto pool3d   = Operator::Get("pool3d");
  Operator temp = *pool3d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), D(8), H(8), W(8), C(3);
  Placeholder<float> A("A", {N, D, H, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size     = {2, 2, 2};
  std::vector<int> stride_size     = {2, 2, 2};
  std::vector<int> padding_size    = {1, 1, 1, 1, 1, 1};
  std::string pool_type            = "max";
  std::string data_format          = "NDHWC";
  attrs.attr_store["kernel_size"]  = kernel_size;
  attrs.attr_store["stride_size"]  = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"]    = pool_type;
  attrs.attr_store["ceil_mode"]    = false;
  attrs.attr_store["exclusive"]    = true;
  attrs.attr_store["data_format"]  = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl =
      OpStrategy::SelectImpl(strategy[pool3d](attrs, inputs, type, {{1, 11, 11, 11, 3}, {1, 5, 5, 5, 3}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("pool3d", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("pool3d");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {1, 8, 8, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {1, 11, 11, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {1, 5, 5, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool3d.x86");
  ASSERT_EQ(pool3d->description, "Do pooling on the depth, height and width dimension of the input tensor.");
}

TEST(Operator, Operator_Pool1d_Test0) {
  auto pool1d   = Operator::Get("pool1d");
  Operator temp = *pool1d;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  Expr N(1), W(8), C(3);
  Placeholder<float> A("A", {N, W, C});

  NodeAttr attrs;
  std::vector<int> kernel_size     = {2};
  std::vector<int> stride_size     = {2};
  std::vector<int> padding_size    = {1, 1};
  std::string pool_type            = "max";
  std::string data_format          = "NWC";
  attrs.attr_store["kernel_size"]  = kernel_size;
  attrs.attr_store["stride_size"]  = stride_size;
  attrs.attr_store["padding_size"] = padding_size;
  attrs.attr_store["pool_type"]    = pool_type;
  attrs.attr_store["ceil_mode"]    = false;
  attrs.attr_store["exclusive"]    = true;
  attrs.attr_store["data_format"]  = data_format;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();
  auto impl = OpStrategy::SelectImpl(strategy[pool1d](attrs, inputs, type, {{1, 11, 3}, {1, 5, 3}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);
  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("pool1d", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("pool1d");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {1, 8, 3}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {1, 11, 3}).set_random().Build();
  cinn_buffer_t *C_buf = common::BufferBuilder(Float(32), {1, 5, 3}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf), c_arg(C_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg, c_arg};
  fn_(args, 3);

  ASSERT_EQ(impl->name, "strategy.pool1d.x86");
  ASSERT_EQ(pool1d->description, "Do pooling on the width dimension of the input tensor.");
}

TEST(Operator, Operator_Reverse_Test0) {
  auto reverse  = Operator::Get("reverse");
  Operator temp = *reverse;
  auto strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  int c = 16, h = 64, w = 64;
  Expr C(c), H(h), W(w);
  Placeholder<float> A("A", {C, H, W});

  NodeAttr attrs;
  std::vector<int> axis    = {1, 2};
  attrs.attr_store["axis"] = axis;
  std::vector<ir::Tensor> inputs{A.tensor()};
  std::vector<Type> type{Float(32)};
  common::Target target = common::DefaultHostTarget();

  auto impl = OpStrategy::SelectImpl(strategy[reverse](attrs, inputs, type, {{c, h, w}}, target));
  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(A)}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);
  ASSERT_EQ(rets.size(), 2UL);

  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("reverse", rets.back(), inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  Module::Builder builder("module0", target);
  builder.AddFunction(func);
  auto jit    = backends::ExecutionEngine::Create({});
  auto module = builder.Build();

  jit->Link(module);
  auto fn = jit->Lookup("reverse");
  CHECK(fn);
  auto fn_ = reinterpret_cast<void (*)(void *, int32_t)>(fn);

  cinn_buffer_t *A_buf = common::BufferBuilder(Float(32), {c, h, w}).set_random().Build();
  cinn_buffer_t *B_buf = common::BufferBuilder(Float(32), {c, h, w}).set_random().Build();
  cinn_pod_value_t a_arg(A_buf), b_arg(B_buf);
  cinn_pod_value_t args[] = {a_arg, b_arg};
  fn_(args, 2);

  auto input  = reinterpret_cast<float *>(A_buf->memory);
  auto output = reinterpret_cast<float *>(B_buf->memory);

  for (int ida = 0; ida < c; ++ida) {
    for (int idb = 0; idb < h; ++idb) {
      for (int idc = 0; idc < w; ++idc) {
        int index  = ida * h * w + idb * h + idc;
        int index_ = ida * h * w + (h - 1 - idb) * h + (w - 1 - idc);
        ASSERT_EQ(output[index], input[index_]);
      }
    }
  }

  ASSERT_EQ(impl->name, "strategy.reverse.x86");
  ASSERT_EQ(reverse->description, "This operator implements the meta op reverse.");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
