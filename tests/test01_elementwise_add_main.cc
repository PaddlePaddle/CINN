#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/optim/optimize.h"
namespace cinn {

TEST(test01_elementwise_add, basic) {
  Expr M(100), N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Buffer C_buf(Float(32));
  auto C = hlir::pe::Add(A.tensor(), B.tensor(), "C");
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module::Builder builder("module1", target);

  auto stages = CreateStages({A, B, C});
  auto func   = Lower("add1", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add.h").c_source("./test01_elementwise_add.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(test01_elementwise_add, vectorize) {
  Expr M(100);
  Expr N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = hlir::pe::Add(A.tensor(), B.tensor(), "C");
  C->stage()->Vectorize(1, 8);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module::Builder builder("module2", target);

  auto stages = CreateStages({A, B, C});
  auto funcs  = Lower("add1_vectorize", stages, {A, B, C});

  auto func = Optimize(funcs);
  LOG(INFO) << "after optim:\n" << func;
  builder.AddFunction(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(C_buf);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_vectorize.h").c_source("./test01_elementwise_add_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

auto BuildComputeAtExpr() {
  Expr M(100);
  Expr N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto A_cache = Compute({M, N}, [=](Expr i, Expr j) {
    auto first = cinn::common::select(i > 0, A(i - 1, j), common::make_const(Float(32), 0.f));
    auto last  = cinn::common::select(i < M - 1, A(i + 1, j), common::make_const(Float(32), 0.f));
    return first + A(i, j) + last;
  });
  auto C       = Compute(
      {M, N}, [&](Var i, Var j) { return A_cache(i, j) + B(i, j); }, "C");

  return std::make_tuple(A, B, A_cache, C);
}

TEST(elementwise_add, compute_at) {
  auto [A, B, A_cache, C] = BuildComputeAtExpr();
  A_cache->stage()->ComputeAt(C->stage(), 0);

  Module::Builder builder("module3", common::DefaultHostTarget());

  auto stages = CreateStages({A, B, C, A_cache});
  auto fn     = Lower("fn_compute_at", stages, {A, B, C}, {}, {A_cache}, &builder);

  CodeGenCX86 compiler(common::DefaultHostTarget(), CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs =
      outputs.c_header("./test01_elementwise_add_compute_at.h").c_source("./test01_elementwise_add_compute_at.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(elementwise_add, compute_at1) {
  auto [A, B, A_cache, C] = BuildComputeAtExpr();
  A_cache->stage()->ComputeAt(C->stage(), 1);

  Module::Builder builder("module4", common::DefaultHostTarget());

  auto stages = CreateStages({A, B, C, A_cache});
  auto fn     = Lower("fn_compute_at_level1", stages, {A, B, C}, {}, {A_cache}, &builder);

  CodeGenCX86 compiler(common::DefaultHostTarget(), CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_compute_at_level1.h")
                .c_source("./test01_elementwise_add_compute_at_level1.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn
