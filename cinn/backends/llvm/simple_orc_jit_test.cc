#include "cinn/backends/llvm/simple_orc_jit.h"
#include "cinn/hlir/instruction/x86/cpu_intrisics.h"

#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <gtest/gtest.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Function.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/cinn.h"
#include "cinn/common/test_helper.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/optimize.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/FileSystem.h"

namespace cinn {
namespace backends {

const int kM = 100;
const int kN = 32;

namespace {
auto CreateTestBuffer() {
  auto *A = common::BufferBuilder(Float(32), {kM, kN}).set_random().Build();
  auto *B = common::BufferBuilder(Float(32), {kM, kN}).set_random().Build();
  auto *C = common::BufferBuilder(Float(32), {kM, kN}).set_zero().Build();
  return std::make_tuple(A, B, C);
}

auto CreateTestCinnModule() {
  ir::Expr M(kM);
  ir::Expr N(kN);
  lang::Placeholder<float> A("A", {M, N});
  lang::Placeholder<float> B("B", {M, N});

  lang::Buffer C_buf(Float(32));
  auto C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);

  common::Target target;
  target.arch = common::Target::Arch ::X86;
  target.bits = common::Target::Bit ::k32;
  target.os   = common::Target::OS ::Linux;
  lang::Module::Builder builder("module1", target);

  auto funcs = lang::Lower("elementwise_add", {A, B, C});

  LOG(INFO) << "funcs:\n" << Expr(funcs);

  builder.AddFunction(ir::LoweredFunc(funcs.As<ir::_LoweredFunc_>()));
  return builder.Build();
}
}  // namespace

TEST(llvm_test01, elementwise_add) {
  auto jit = backends::SimpleOrcJit::Create();

  auto [a, b, c] = CreateTestBuffer();  // NOLINT

  auto module = CreateTestCinnModule();

  jit->Link(module, /*optimize=*/true);

  auto elementwise_add_addr = jit->Lookup("elementwise_add");
  auto elementwise_add      = reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);
  CHECK(elementwise_add);
  cinn_pod_value_t a_arg(a), b_arg(b), c_arg(c);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};
  elementwise_add(args, 3);

  float *ad = reinterpret_cast<float *>(a->host_memory);
  float *bd = reinterpret_cast<float *>(b->host_memory);
  float *cd = reinterpret_cast<float *>(c->host_memory);

  for (int i = 0; i < c->num_elements(); i++) {
    EXPECT_EQ(ad[i] + bd[i], cd[i]);
  }
}

TEST(llvm, module_call_lowered_func) {
  lang::Module::Builder builder("some_module", common::DefaultHostTarget());
  ir::Expr M(kM);
  ir::Expr N(kN);
  {  // define fn
    lang::Placeholder<float> a("a", {M, N});
    lang::Placeholder<float> b("b", {M, N});
    auto c = lang::Compute(
        {M, N}, [&](auto i, auto j) { return a(i, j) + b(i, j); }, "c");
    c->WithBuffer();

    auto fn = lang::Lower("elementwise_add", {a, b, c}, {});
    builder.AddFunction(fn);
  }

  {  // call fn
    lang::Placeholder<float> a("a", {M, N});
    lang::Placeholder<float> b("b", {M, N});

    std::vector<lang::ReturnType> ret_types({lang::ReturnType{Float(32), {M, N}, "c_out"}});

    auto call_outs = lang::Call("elementwise_add", {a, b}, ret_types);
    auto c         = call_outs[0];

    // here we must call the output, so that it cal output something.

    auto main_fn = lang::Lower("main", {a, b, c}, {});
    builder.AddFunction(main_fn);
  }

  auto [ab, bb, cb] = CreateTestBuffer();  // NOLINT
  do {                                     // call the function
    auto jit = backends::SimpleOrcJit::Create();

    LOG(INFO) << "JIT Link the module";
    jit->Link(builder.Build(), /*optimize=*/false);
    auto elementwise_add_addr = jit->Lookup("elementwise_add");
    auto elementwise_add      = reinterpret_cast<void (*)(void *, int32_t)>(elementwise_add_addr);
    LOG(INFO) << "JIT get elementwise_add_addr";

    cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
    cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

    elementwise_add(args, 3);

    auto *ad = reinterpret_cast<float *>(ab->host_memory);
    auto *bd = reinterpret_cast<float *>(bb->host_memory);
    for (int i = 0; i < kM; i++) {
      for (int j = 0; j < kN; j++) {
        auto *data = reinterpret_cast<float *>(cb->host_memory);
        ASSERT_NEAR(data[i * kN + j], ad[i * kN + j] + bd[i * kN + j], 1e-5);
      }
    }
  } while (false);
}

TEST(jit, cpu_runtime) {
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module  = std::make_unique<llvm::Module>("test_llvm_cpu_runtime", *context);
  auto builder = std::make_unique<llvm::IRBuilder<>>(*context);

  auto call_custom_target = [&](std::string name, llvm::Type *ty) {
    llvm::FunctionType *fn_type = llvm::FunctionType::get(ty, {ty}, false);
    llvm::Function *function =
        llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, "_call_custom_" + name, module.get());
    function->setCallingConv(llvm::CallingConv::C);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(module->getContext(), "entry", function);
    builder->SetInsertPoint(entry);
    llvm::Argument *arg = &*function->args().begin();
    llvm::Function *custom_function =
        llvm::dyn_cast<llvm::Function>(module->getOrInsertFunction(name, fn_type).getCallee());
    custom_function->setCallingConv(llvm::CallingConv::C);
    llvm::Value *ret = builder->CreateCall(custom_function, {arg});
    builder->CreateRet(ret);
  };

  llvm::Type *f32 = builder->getFloatTy();
  llvm::Type *f64 = builder->getDoubleTy();
  call_custom_target("cosf", f32);
  call_custom_target("cos", f64);
  call_custom_target("sinf", f32);
  call_custom_target("sin", f64);

  auto jit = cinn::backends::SimpleOrcJit::Create();
  jit->AddModule(std::move(module), false);

  double pi = std::acos(-1);

  auto *call_cosf = reinterpret_cast<float (*)(float)>(jit->Lookup("_call_custom_cosf"));
  auto *call_cos  = reinterpret_cast<double (*)(double)>(jit->Lookup("_call_custom_cos"));
  auto *call_sinf = reinterpret_cast<float (*)(float)>(jit->Lookup("_call_custom_sinf"));
  auto *call_sin  = reinterpret_cast<double (*)(double)>(jit->Lookup("_call_custom_sin"));

  ASSERT_TRUE(call_cosf && call_cos && call_sinf && call_sin);

  for (auto theta : {0., pi / 6., pi / 4., pi / 3., pi / 2., pi}) {
    float theta_f = static_cast<float>(theta);
    ASSERT_NEAR(call_cosf(theta_f), cosf(theta_f), 1e-6);
    ASSERT_NEAR(call_cos(theta), cos(theta), 1e-6);
    ASSERT_NEAR(call_sinf(theta_f), sinf(theta_f), 1e-6);
    ASSERT_NEAR(call_sin(theta), sin(theta), 1e-6);
  }
}

TEST(SimpleOrcJit, call_extern) {
  ir::Expr M(kM);
  ir::Expr N(kN);

  Placeholder<float> x("x", {M, N});
  Placeholder<float> y("y", {M, N});

  auto add_out = Compute(
      {M, N}, [=](Var i, Var j) { return x(i, j) + y(i, j); }, "add_out");
  ir::Tensor res = Compute(
      {M, N}, [&](Var i, Var j) -> Expr { return lang::CallExtern("tanh", {add_out(i, j)}); }, "res");
  res->WithBuffer();

  auto func = Lower("comp", {x, y, res});

  Module::Builder builder("module0", common::DefaultHostTarget());
  builder.AddFunction(func);

  auto jit = backends::SimpleOrcJit::Create();

  LOG(INFO) << "JIT Link the module";
  jit->Link(builder.Build(), /*optimize=*/false);

  auto [ab, bb, cb] = CreateTestBuffer();  // NOLINT

  auto comp_addr = jit->Lookup("comp");
  auto comp      = reinterpret_cast<void (*)(void *, int32_t)>(comp_addr);

  cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

  comp(args, 3);

  auto *ad = reinterpret_cast<float *>(ab->host_memory);
  auto *bd = reinterpret_cast<float *>(bb->host_memory);
  auto *cd = reinterpret_cast<float *>(cb->host_memory);
  for (int m = 0; m < kM; m++) {
    for (int n = 0; n < kN; n++) {
      ASSERT_NEAR(cd[m * kN + n], __cinn_host_tanh_fp32(ad[m * kN + n] + bd[m * kN + n]), 1e-5);
    }
  }
}

TEST(SimpleOrcJit, call_extern_v_generate) {
  ir::Expr M(kM);
  ir::Expr N(kN);

  Placeholder<float> x("x", {M});

  auto y = Compute(
      {Expr(1)}, [&]() -> Expr { return lang::CallExtern("tanh_v", {x}); }, "out");

  auto y1 = Compute(
      {M}, [&](Var i) -> Expr { return lang::CallExtern("tanh_v", {x}); }, "out1");

  auto yy = y->TupleGet(0);
  yy->WithBuffer();

  auto yy1 = y1->TupleGet(0);
  yy1->WithBuffer();

  auto func = Lower("comp", {x, y, y1, yy, yy1});

  LOG(INFO) << "func: " << func;
}

TEST(SimpleOrcJit, call_extern_tanh_v) {
  ir::Expr M(kM);
  ir::Expr N(kN);

  Placeholder<float> x("x", {M, N});
  Placeholder<float> y("y", {M, N});

  auto add_out = Compute(
      {M, N}, [=](Var i, Var j) { return x(i, j) + y(i, j); }, "add_out");

  auto tanh_out = Compute({M, N}, [=](Var i, Var j) { return lang::CallExtern("tanh", {add_out(i, j)}); });
  tanh_out->WithBuffer();

  auto func = Lower("tanh_main", {x, y, tanh_out});

  Module::Builder builder("module0", common::DefaultHostTarget());
  builder.AddFunction(func);

  auto jit = backends::SimpleOrcJit::Create();

  LOG(INFO) << "JIT Link the module";
  jit->Link(builder.Build(), /*optimize=*/false);

  auto [ab, bb, cb] = CreateTestBuffer();  // NOLINT

  auto comp_addr = jit->Lookup("tanh_main");
  auto comp      = reinterpret_cast<void (*)(void *, int32_t)>(comp_addr);

  cinn_pod_value_t a_arg(ab), b_arg(bb), c_arg(cb);
  cinn_pod_value_t args[3] = {a_arg, b_arg, c_arg};

  comp(args, 3);

  auto *ad = reinterpret_cast<float *>(ab->host_memory);
  auto *bd = reinterpret_cast<float *>(bb->host_memory);
  auto *cd = reinterpret_cast<float *>(cb->host_memory);
  for (int m = 0; m < kM; m++) {
    for (int n = 0; n < kN; n++) {
      ASSERT_NEAR(cd[m * kN + n], __cinn_host_tanh_fp32(ad[m * kN + n] + bd[m * kN + n]), 1e-5);
    }
  }
}

TEST(SimpleOrcJit, call_extern_v) {
  ir::Expr M(kM);
  ir::Expr N(kN);

  char *fn_name = "main";

  Placeholder<float> x("x", {M, N});

  auto y = Compute(
      {Expr(1)}, [&]() -> Expr { return lang::CallExtern("tanh_v", {x}); }, "out");

  auto yy = y->TupleGet(0);
  yy->WithBuffer(Float(32));

  lang::Module::Builder builder("module0", common::DefaultHostTarget());

  auto func = Lower(fn_name,
                    {
                        x,
                        y,
                        yy,
                    });

  builder.AddFunction(func);

  auto jit = backends::SimpleOrcJit::Create();

  LOG(INFO) << "JIT Link the module";
  jit->Link(builder.Build(), /*optimize=*/false);

  auto [ab, bb, cb] = CreateTestBuffer();  // NOLINT

  auto comp_addr = jit->Lookup(fn_name);
  auto comp      = reinterpret_cast<void (*)(void *, int32_t)>(comp_addr);
  CHECK(comp);

  cinn_pod_value_t a_arg(ab), b_arg(bb);
  cinn_pod_value_t args[3] = {a_arg, b_arg};

  comp(args, 2);

  auto *ad = reinterpret_cast<float *>(ab->host_memory);
  auto *bd = reinterpret_cast<float *>(bb->host_memory);
  for (int i = 0; i < bb->num_elements(); i++) {
    ASSERT_NEAR(bd[i], __cinn_host_tanh_fp32(ad[i]), 1e-5);
  }
}

// Change the shape
TEST(SimpleOrcJit, shape_view) {
  ir::Expr M0(10);
  ir::Expr M1(20);
  ir::Expr M2(30);

  std::vector<Expr> shape({M0, M1, M2});
  Placeholder<float> x("x", shape);

  auto y = Compute({M0 * M1, M2}, [&](Var i, Var j) -> Expr { return x(i / M1, i % M1, j); }, "y", {});
  y->Bind(ir::Tensor(x)->buffer);

  auto z = Compute(
      {M0 * M1, M2}, [&](Var i, Var j) { return y(i, j) + 1.f; }, "z");
  z->Bind(y->buffer);

  auto func = Lower("fn", {x, y, z});
  LOG(INFO) << "func:\n" << func;
}

}  // namespace backends
}  // namespace cinn
