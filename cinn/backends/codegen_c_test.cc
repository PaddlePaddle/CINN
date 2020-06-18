#include "cinn/backends/codegen_c.h"

#include <gtest/gtest.h>

#include <sstream>
#include <tuple>

#include "cinn/ir/ir.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace backends {

using lang::Compute;
using lang::Lower;
using lang::Module;
using lang::Placeholder;
using utils::StringFormat;
using utils::Trim;

std::tuple<ir::Tensor, ir::Tensor, ir::Tensor, lang::Buffer> CreateTensor1() {
  Expr M(100);
  Expr N(20);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  lang::Buffer C_buf(Float(32));
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);
  return std::make_tuple(A, B, C, C_buf);
}

TEST(CodeGenC, module) {
  ir::Tensor A, B, C;
  lang::Buffer C_buf(Float(32));
  std::tie(A, B, C, C_buf) = CreateTensor1();

  LOG(INFO) << "C.body: " << C->get_compute_op()->body.front();

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module::Builder builder("module1", target);

  auto func = Lower("add1", {A, B, C});

  builder.AddFunction(func);
  builder.AddBuffer(C_buf.buffer());

  {
    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "codegen C:" << std::endl << out << std::endl;

    std::string target_str = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 20 }, 32/*align*/);
void add1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 20; j += 1) {
      C[((20 * i) + j)] = (A[((20 * i) + j)] + B[((20 * i) + j)]);
    };
  };
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
    EXPECT_EQ(utils::Trim(target_str), utils::Trim(out));
  }

  {
    CodeGenC compiler(target);
    auto out = compiler.Compile(builder.Build(), CodeGenC::OutputKind::CHeader);
    std::cout << "header:\n" << out << std::endl;
    auto target_str = R"ROC(
#ifndef _MODULE1_CINN_H_
#define _MODULE1_CINN_H_

#include <cinn_runtime.h>
#include <stdio.h>

void add1(void* _args, int32_t num_args);


#endif  // _MODULE1_CINN_H_
)ROC";

    EXPECT_EQ(utils::Trim(out), utils::Trim(target_str));
  }

  {
    CodeGenC compiler(target);
    compiler.SetInlineBuiltinCodes(false);
    Outputs outputs;
    outputs = outputs.c_header("./generated_module1.h").c_source("./generated_module1.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(CodeGenC, module_with_transform) {
  Expr M(100);
  Expr N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  lang::Buffer C_buf(Float(32)), D_buf(Float(32));

  // An inlined tensor, should not appear in final C code! It can be used by any times and expand its expression there.
  auto inlined0 = Compute({M, N}, [&](Var i, Var j) { return A(i, j) * 2.f + 1.f; });

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j) + inlined0(i, j); }, "C");
  C->Bind(C_buf);

  auto D = Compute(
      {M, N}, [&](Var i, Var j) { return C(i, j) * 2.f * inlined0(i, j); }, "D");
  D->Bind(D_buf);

  poly::Iterator i_outer, i_inner;
  std::tie(i_outer, i_inner) = C->stage()->Split(poly::DefaultIterator(0), 4);

  D->stage()->Tile(poly::DefaultIterator(0), poly::DefaultIterator(1), 4, 16);

  Target target = common::DefaultHostTarget();
  Module::Builder builder("module1", target);

  auto funcs = Lower("add1", {A, B, C, D});

  Expr func(funcs);
  optim::Simplify(&func);

  builder.AddFunction(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  builder.AddBuffer(C_buf.buffer());

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto tgt = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 20 }, 32/*align*/);
void add1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* _D = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[3]));
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _D);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  float* D = ((float*)(_D->host_memory));
  for (int32_t i_outer = 0; i_outer < 25; i_outer += 1) {
    for (int32_t i_inner = 0; i_inner < 4; i_inner += 1) {
      for (int32_t j = 0; j < 20; j += 1) {
        C[((20 * i_inner) + ((80 * i_outer) + j))] = (1 + ((3 * A[((20 * i_inner) + ((80 * i_outer) + j))]) + B[((20 * i_inner) + ((80 * i_outer) + j))]));
      };
    };
  };
  for (int32_t i_outer = 0; i_outer < 25; i_outer += 1) {
    for (int32_t i_inner = 0; i_inner < 4; i_inner += 1) {
      for (int32_t j_outer = 0; j_outer < 1; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < 16; j_inner += 1) {
          D[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))] = ((2 * C[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))]) + (4 * (C[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))] * A[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))])));
        };
      };
      for (int32_t j_outer = 1; j_outer < 2; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < (20 + (-16 * j_outer)); j_inner += 1) {
          D[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))] = ((2 * C[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))]) + (4 * (C[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))] * A[((20 * i_inner) + ((80 * i_outer) + ((16 * j_outer) + j_inner)))])));
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _C);
  cinn_buffer_free((void*)(0), _D);
}
)ROC";

  ASSERT_EQ(utils::Trim(tgt), utils::Trim(out));
}

TEST(CodeGenC, matmul) {
  using namespace ir;  // NOLINT

  Placeholder<float> A("A", {Expr(100), Expr(20)});
  Placeholder<float> B("B", {Expr(20), Expr(50)});

  Target target{};

  Module::Builder builder("module1", target);

  // C = A * B
  Var k(20, "k0");

  Tensor C_init = Compute(
      {Expr(100), Expr(50)}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();

  Tensor C = Compute({Expr(100), Expr(50)}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);

  C_init->stage()->ComputeAt(C->stage(), 1, poly::Stage::kComputeAtBefore);

  // Code gen
  auto func = Lower("matmul", {A, B, C_init, C});
  builder.AddFunction(func);
  builder.AddBuffer(C->buffer);

  {  // main
    std::vector<lang::ReturnType> returns({lang::ReturnType{Float(32), C->shape, C->name}});

    auto tensors = lang::Call("matmul", {A, B}, returns);

    auto C = tensors[0];
    C->WithBuffer();

    LOG(INFO) << "C.body: " << C->body();

    auto f = Lower("main", {A, B, C}, {});
    std::cout << "f\n" << Expr(f) << std::endl;
    builder.AddFunction(f);
  }

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto tgt = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C_init = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 50 });
void matmul(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C_init = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C_init);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C_init->host_memory));
  float* C_init = ((float*)(_C_init->host_memory));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 50; j += 1) {
      C_init[((50 * i) + j)] = 0;
      for (int32_t k0 = 0; k0 < 20; k0 += 1) {
        C[((50 * i) + j)] = (C[((50 * i) + j)] + (A[((20 * i) + k0)] * B[((50 * k0) + j)]));
      };
    };
  };
  cinn_buffer_free((void*)(0), _C_init);
}

void main(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  {
    cinn_pod_value_t _pod_val__8;
    buffer_p_to_cinn_pod_value(_A, &_pod_val__8);
    cinn_pod_value_t _pod_val__9;
    buffer_p_to_cinn_pod_value(_B, &_pod_val__9);
    cinn_pod_value_t _pod_val__10;
    buffer_p_to_cinn_pod_value(_C, &_pod_val__10);
    cinn_pod_value_t _pod_arr_11[3];
    cinn_args_construct(_pod_arr_11, 3, &_pod_val__8, &_pod_val__9, &_pod_val__10);
    matmul(_pod_arr_11, 5);
  };
  cinn_buffer_free((void*)(0), _C);
}
)ROC";

  ASSERT_EQ(Trim(tgt), Trim(out));
}

// This matches output of competitor.
TEST(CodeGenC, matmul_tile) {
  using namespace ir;  // NOLINT
  Expr M(100);
  Expr K(200);
  Expr N(500);
  Expr bn(32);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));

  Var k(K.as_int32(), "k0");

  Tensor C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");
  C_init->WithBuffer();

  Tensor C = Compute({M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_init->buffer);
  // C_init->stage()->ComputeAt(C->stage(), 1, poly::Stage::kComputeAtBefore);

  {
    auto [i_outer, i_inner, j_outer, j_inner] = C_init->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());  // NOLINT
    C_init->stage()->Reorder({i_outer, j_outer, i_inner, j_inner});
  }

  {
    auto [i_outer, i_inner, j_outer, j_inner] = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());  // NOLINT
    auto [k_outer, k_inner]                   = C->stage()->Split(poly::Iterator("k0"), 4);            // NOLINT
    C->stage()->Reorder({i_outer, j_outer, i_inner, j_inner, k_outer, k_inner});
  }

  C_init->stage()->ComputeAt(C->stage(), 3, poly::Stage::kComputeAtBefore);

  // Code gen
  auto func = Lower("matmul", {A, B, C_init, C});

  Target target = common::DefaultHostTarget();

  Module::Builder builder("module1", target);
  builder.AddFunction(func);
  builder.AddBuffer(C_init->buffer);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C_init = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 }, 32/*align*/);
void matmul(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C_init = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C_init);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C_init->host_memory));
  float* C_init = ((float*)(_C_init->host_memory));
  for (int32_t i_outer = 0; i_outer < 3; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < 32; j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * B[((32 * j_outer) + ((500 * k0_inner) + ((2000 * k0_outer) + j_inner)))]));
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * B[((32 * j_outer) + ((500 * k0_inner) + ((2000 * k0_outer) + j_inner)))]));
            };
          };
        };
      };
    };
  };
  for (int32_t i_outer = 3; i_outer < 4; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < 32; j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * B[((32 * j_outer) + ((500 * k0_inner) + ((2000 * k0_outer) + j_inner)))]));
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * B[((32 * j_outer) + ((500 * k0_inner) + ((2000 * k0_outer) + j_inner)))]));
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _C_init);
}
)ROC";

  ASSERT_EQ(Trim(target_out), Trim(out));
}

TEST(CodeGenC, matmul_packed) {
  Expr M(100);
  Expr K(200);
  Expr N(500);
  Expr bn(32);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  lang::Buffer packedB_buf(Float(32));
  lang::Buffer C_buf(Float(32));

  // TODO(Superjomn) Make sure the domain works.
  Var k(K.as_int32(), "k0");
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "PackedB");
  packedB->Bind(packedB_buf);
  auto C = Compute({M, N}, [&](Expr i, Expr j) { return A(i, k) * packedB(j / bn, k, j % bn); }, "C", {k});
  C->Bind(C_buf);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k0"), 4);
    C->stage()->Reorder({i_outer, j_outer, i_inner, j_inner, k_outer, k_inner});
  }

  // Code gen
  auto func = Lower("matmul_with_packing", {A, B, packedB, C});

  Target target = common::DefaultHostTarget();

  Module::Builder builder("module1", target);
  builder.AddFunction(func);
  builder.AddBuffer(C_buf.buffer());
  builder.AddBuffer(packedB_buf.buffer());

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 }, 32/*align*/);
cinn_buffer_t* _PackedB = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 15, 200, 32 }, 32/*align*/);
void matmul_with_packing(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _PackedB = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[3]));
  cinn_buffer_malloc((void*)(0), _PackedB);
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  float* PackedB = ((float*)(_PackedB->host_memory));
  for (int32_t i = 0; i < 15; i += 1) {
    for (int32_t j = 0; j < 200; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        PackedB[((6400 * i) + ((32 * j) + k))] = B[((32 * i) + ((500 * j) + k))];
      };
    };
  };
  for (int32_t i_outer = 0; i_outer < 3; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < 32; j_inner += 1) {
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * PackedB[((6400 * j_outer) + ((32 * k0_inner) + ((128 * k0_outer) + j_inner)))]);
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * PackedB[((j_inner % 32) + ((6400 * (j_inner/32)) + ((6400 * j_outer) + ((32 * k0_inner) + (128 * k0_outer)))))]);
            };
          };
        };
      };
    };
  };
  for (int32_t i_outer = 3; i_outer < 4; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < 32; j_inner += 1) {
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * PackedB[((6400 * j_outer) + ((32 * k0_inner) + ((128 * k0_outer) + j_inner)))]);
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          for (int32_t k0_outer = 0; k0_outer < 50; k0_outer += 1) {
            for (int32_t k0_inner = 0; k0_inner < 4; k0_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k0_outer) + k0_inner)))] * PackedB[((j_inner % 32) + ((6400 * (j_inner/32)) + ((6400 * j_outer) + ((32 * k0_inner) + (128 * k0_outer)))))]);
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _PackedB);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";

  ASSERT_EQ(utils::Trim(out), utils::Trim(target_out));
}

TEST(CodeGenC, call_extern) {
  Expr M(100);

  Placeholder<float> x("x", {M});

  ir::Tensor y = Compute(
      {M}, [=](Var i) -> Expr { return lang::CallExtern("tanh", {x(i)}); }, "y");
  y->WithBuffer();

  auto yexpr = Lower("yy", {y});

  Module::Builder builder("module0", common::DefaultHostTarget());
  builder.AddFunction(yexpr);

  CodeGenC codegen(common::DefaultHostTarget());
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;
}

TEST(CodeGenC, cache_read) {
  Expr M(100), N(200);

  Placeholder<float> A("A", {M, N});

  // 1. original compute way
  auto original_compute = Compute(
      {M, N}, [&](Var i, Var j) { return ir::Select::Make(j > 1, A(i, j) + A(i, j - 1), A(i, j)); }, "origin");
  original_compute->WithBuffer();

  // 2. cached compute way
  auto cache_prepare = Compute({M, N} /*domain*/, [&](Var i, Var j) { return A(i, j); }, "cache", {}, {N} /*shape*/);
  cache_prepare->WithBuffer();

  auto transformed_compute = Compute(
      {M, N}, [&](Var i, Var j) { return cache_prepare(j); }, "transformed");
  transformed_compute->WithBuffer();

  cache_prepare->stage()->ComputeAt(transformed_compute->stage(), 1);

  // codegen and compare
  auto fn = Lower("fn", {A, original_compute, cache_prepare, transformed_compute});

  LOG(INFO) << "fn:\n" << fn;

  ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), utils::Trim(R"ROC(
function fn (_A, _origin, _cache, _transformed)
{
  for (i, 100)
  {
    for (j, 200)
    {
      origin[i, j] = select((j > 1), (A[i, j] + A[i, (-1 + j)]), A[i, j])
    }
  }
  for (i, 100)
  {
    for (j, 200)
    {
      cache[i] = A[i, j]
      transformed[i, j] = cache[j]
    }
  }
}
)ROC"));
}

}  // namespace backends
}  // namespace cinn
