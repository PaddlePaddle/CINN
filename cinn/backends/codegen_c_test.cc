#include "cinn/backends/codegen_c.h"

#include <gtest/gtest.h>

#include <sstream>
#include <tuple>

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
  Module module("module1", target);

  auto func = Lower("add1", {A, B, C});

  module.Append(func);
  module.Append(C_buf);

  {
    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
    std::cout << "codegen C:" << std::endl << out << std::endl;

    std::string target_str = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 20 });
void add1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[2]));
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
    auto out = compiler.Compile(module, CodeGenC::OutputKind::CHeader);
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
    Outputs outputs;
    outputs = outputs.c_header("./generated_module1.h").c_source("./generated_module1.cc");
    compiler.Compile(module, outputs);
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

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  Module module("module1", target);

  auto funcs = Lower("add1", {A, B, C, D});

  Expr func(funcs);
  optim::Simplify(&func);

  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  module.Append(C_buf);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto tgt = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 20 });
void add1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* _D = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[3]));
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

  // C = A * B
  lang::Buffer C_buf(Float(32));
  Var k(20, "k");

  Tensor C_init = Compute(
      {Expr(100), Expr(50)}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");

  Tensor C = Compute({Expr(100), Expr(50)}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_buf);
  C_init->Bind(C_buf);
  C_init->stage()->ComputeAt(C->stage(), 1);

  // Code gen
  auto func = Lower("matmul", {A, B, C_init, C});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Module module("module1", target);
  module.Append(func);
  module.Append(C_buf);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto tgt = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 50 });
void matmul(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  float* C_init = ((float*)(_C->host_memory));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 50; j += 1) {
      C_init[((50 * i) + j)] = 0;
      for (int32_t k = 0; k < 20; k += 1) {
        C[((50 * i) + j)] = (C[((50 * i) + j)] + (A[((20 * i) + k)] * B[((50 * k) + j)]));
      };
    };
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
  Var k(K.as_int32(), "k");

  Tensor C_init = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(0.f); }, "C_init");

  Tensor C = Compute({M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j)); }, "C", {k});
  C->Bind(C_buf);
  C_init->Bind(C_buf);
  // C_init->stage()->ComputeAt(C->stage(), 1);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C_init->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    C_init->stage()->Reorder({i_outer, j_outer, i_inner, j_inner});
  }

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k"), 4);
    C->stage()->Reorder({i_outer, j_outer, i_inner, j_inner, k_outer, k_inner});
  }

  C_init->stage()->ComputeAt(C->stage(), 3);

  // Code gen
  auto func = Lower("matmul", {A, B, C_init, C});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Module module("module1", target);
  module.Append(func);
  module.Append(C_buf);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 });
void matmul(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->host_memory));
  const float* B = ((const float*)(_B->host_memory));
  float* C = ((float*)(_C->host_memory));
  float* C_init = ((float*)(_C->host_memory));
  for (int32_t i_outer = 0; i_outer < 3; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < 32; j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * B[((32 * j_outer) + ((500 * k_inner) + ((2000 * k_outer) + j_inner)))]));
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * B[((32 * j_outer) + ((500 * k_inner) + ((2000 * k_outer) + j_inner)))]));
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
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * B[((32 * j_outer) + ((500 * k_inner) + ((2000 * k_outer) + j_inner)))]));
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          C_init[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = 0;
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * B[((32 * j_outer) + ((500 * k_inner) + ((2000 * k_outer) + j_inner)))]));
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _C);
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
  Var k(K.as_int32(), "k");
  auto packedB = Compute(
      {N / bn, K, bn}, [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); }, "PackedB");
  packedB->Bind(packedB_buf);
  auto C = Compute({M, N}, [&](Expr i, Expr j) { return A(i, k) * packedB(j / bn, k, j % bn); }, "C", {k});
  C->Bind(C_buf);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn.as_int32(), bn.as_int32());
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k"), 4);
    C->stage()->Reorder({i_outer, j_outer, i_inner, j_inner, k_outer, k_inner});
  }

  // Code gen
  auto func = Lower("matmul_with_packing", {A, B, packedB, C});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Module module("module1", target);
  module.Append(func);
  module.Append(C_buf);
  module.Append(packedB_buf);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 });
cinn_buffer_t* _PackedB = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 15, 200, 32 });
void matmul_with_packing(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = ((const cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _PackedB = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* _C = ((cinn_buffer_t*)(((cinn_pod_value_t*)(_args))[3]));
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
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * PackedB[((6400 * j_outer) + ((32 * k_inner) + ((128 * k_outer) + j_inner)))]);
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * PackedB[((j_inner % 32) + ((6400 * (j_inner/32)) + ((6400 * j_outer) + ((32 * k_inner) + (128 * k_outer)))))]);
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
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * PackedB[((6400 * j_outer) + ((32 * k_inner) + ((128 * k_outer) + j_inner)))]);
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
        for (int32_t j_inner = 0; j_inner < (500 + (-32 * j_outer)); j_inner += 1) {
          for (int32_t k_outer = 0; k_outer < 50; k_outer += 1) {
            for (int32_t k_inner = 0; k_inner < 4; k_inner += 1) {
              C[((500 * i_inner) + ((16000 * i_outer) + ((32 * j_outer) + j_inner)))] = (A[((200 * i_inner) + ((6400 * i_outer) + ((4 * k_outer) + k_inner)))] * PackedB[((j_inner % 32) + ((6400 * (j_inner/32)) + ((6400 * j_outer) + ((32 * k_inner) + (128 * k_outer)))))]);
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

  ASSERT_EQ(utils::Trim(target_out), utils::Trim(out));
}

}  // namespace backends
}  // namespace cinn
