#include "cinn/optim/vectorize_loops.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/common/common.h"
#include "cinn/common/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/optimize.h"
#include "cinn/optim/transform_polyfor_to_for.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using utils::GetStreamCnt;
using utils::Trim;

TEST(VectorizeLoops, Split_sperate) {
  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));
  Var k(K, "k");

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return lang::Sum(A(i, k) * B(k, j), k); }, "C", k);
  C->Bind(C_buf);

  {
    poly::Iterator i_outer, i_inner, j_outer, j_inner, k_outer, k_inner;
    std::tie(i_outer, i_inner, j_outer, j_inner) = C->stage()->Tile(0, 1, bn, bn);
    std::tie(k_outer, k_inner)                   = C->stage()->Split(poly::Iterator("k"), 8);
    C->stage()->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    C->stage()->Split(j_inner, 8);
  }

  // Code gen
  auto funcs = Lower("matmul", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Expr body = optim::Optimize(Expr(funcs[0]));

  lang::Module module("module1", target);
  module.Append(ir::LoweredFunc(body.As<ir::_LoweredFunc_>()));
  module.Append(C_buf);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* _C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 100, 500 });
void matmul(const struct cinn_buffer_t *_A, const struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(cinn_buffer_get_data_const_handle(_A));
  const float* B = (const float*)(cinn_buffer_get_data_const_handle(_B));
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  for (int32_t i_outer = 0; i_outer < 3; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t k_outer = 0; k_outer < 25; k_outer += 1) {
        for (int32_t k_inner = 0; k_inner < 8; k_inner += 1) {
          for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
            for (int32_t j_inner_outer = 0; j_inner_outer < 4; j_inner_outer += 1) {
              for (int32_t j_inner_inner = 0; j_inner_inner < min(8, (500 + ((-8 * j_inner_outer) + (-32 * j_outer)))); j_inner_inner += 1) {
                C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((8 * k_outer) + k_inner)))] * B[((8 * j_inner_outer) + ((32 * j_outer) + ((500 * k_inner) + ((4000 * k_outer) + j_inner_inner))))]));
              };
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t k_outer = 0; k_outer < 25; k_outer += 1) {
        for (int32_t k_inner = 0; k_inner < 8; k_inner += 1) {
          for (int32_t i_inner = 0; i_inner < 32; i_inner += 1) {
            for (int32_t j_inner_outer = 0; j_inner_outer < (63 + (-4 * j_outer)); j_inner_outer += 1) {
              for (int32_t j_inner_inner = 0; j_inner_inner < min(8, (500 + ((-8 * j_inner_outer) + (-32 * j_outer)))); j_inner_inner += 1) {
                C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((8 * k_outer) + k_inner)))] * B[((8 * j_inner_outer) + ((32 * j_outer) + ((500 * k_inner) + ((4000 * k_outer) + j_inner_inner))))]));
              };
            };
          };
        };
      };
    };
  };
  for (int32_t i_outer = 3; i_outer < 4; i_outer += 1) {
    for (int32_t j_outer = 0; j_outer < 15; j_outer += 1) {
      for (int32_t k_outer = 0; k_outer < 25; k_outer += 1) {
        for (int32_t k_inner = 0; k_inner < 8; k_inner += 1) {
          for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
            for (int32_t j_inner_outer = 0; j_inner_outer < 4; j_inner_outer += 1) {
              for (int32_t j_inner_inner = 0; j_inner_inner < min(8, (500 + ((-8 * j_inner_outer) + (-32 * j_outer)))); j_inner_inner += 1) {
                C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((8 * k_outer) + k_inner)))] * B[((8 * j_inner_outer) + ((32 * j_outer) + ((500 * k_inner) + ((4000 * k_outer) + j_inner_inner))))]));
              };
            };
          };
        };
      };
    };
    for (int32_t j_outer = 15; j_outer < 16; j_outer += 1) {
      for (int32_t k_outer = 0; k_outer < 25; k_outer += 1) {
        for (int32_t k_inner = 0; k_inner < 8; k_inner += 1) {
          for (int32_t i_inner = 0; i_inner < (100 + (-32 * i_outer)); i_inner += 1) {
            for (int32_t j_inner_outer = 0; j_inner_outer < (63 + (-4 * j_outer)); j_inner_outer += 1) {
              for (int32_t j_inner_inner = 0; j_inner_inner < min(8, (500 + ((-8 * j_inner_outer) + (-32 * j_outer)))); j_inner_inner += 1) {
                C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] = (C[((500 * i_inner) + ((16000 * i_outer) + ((8 * j_inner_outer) + ((32 * j_outer) + j_inner_inner))))] + (A[((200 * i_inner) + ((6400 * i_outer) + ((8 * k_outer) + k_inner)))] * B[((8 * j_inner_outer) + ((32 * j_outer) + ((500 * k_inner) + ((4000 * k_outer) + j_inner_inner))))]));
              };
            };
          };
        };
      };
    };
  };
}
)ROC";

  std::cout << "\n" << out << std::endl;
  EXPECT_EQ(utils::Trim(target_out), utils::Trim(out));
}

TEST(Vectorize, replace_var) {
  using namespace ir;  // NOLINT

  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->Bind(C_buf);

  C->stage()->Vectorize(1, 16);

  auto funcs = Lower("matmul", {A, B, C});
  CHECK_EQ(funcs.size(), 1UL);

  Expr func = optim::Optimize(funcs.front());

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  lang::Module module("module1", target);
  module.Append(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out        = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void matmul(const struct cinn_buffer_t *_A, const struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(cinn_buffer_get_data_const_handle(_A));
  const float* B = (const float*)(cinn_buffer_get_data_const_handle(_B));
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < (125/4); j += 1) {
      C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) + (16 * j))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j))));
    };
  };
}
)ROC";
  EXPECT_EQ(Trim(target_out), Trim(out));
}

TEST(Vectorize, TestMarkVectorize) {
  // create two forloops, check only one forloop is marked Vectorize.
  Context::Global().info_rgt().Clear();

  using namespace ir;  // NOLINT

  const int M  = 100;
  const int K  = 200;
  const int N  = 500;
  const int bn = 32;

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B
  lang::Buffer C_buf(Float(32));

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->Bind(C_buf);

  Tensor D = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "D");
  D->Bind(C_buf);

  // vectorize C, not D
  C->stage()->Vectorize(1, 16);

  auto funcs = Lower("matmul", {A, B, C, D});
  CHECK_EQ(funcs.size(), 1UL);

  std::cout << "before optim\n" << funcs.front()->body << std::endl;

  optim::TransformPolyForToFor(&funcs[0]->body);
  optim::VectorizeLoops(&funcs[0]->body, target);
  optim::Simplify(&funcs[0]->body);

  lang::Module module("module1", target);
  module.Append(funcs[0]);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void matmul(const struct cinn_buffer_t *_A, const struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(cinn_buffer_get_data_const_handle(_A));
  const float* B = (const float*)(cinn_buffer_get_data_const_handle(_B));
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  float* D = (float*)(cinn_buffer_get_data_handle(_C));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j_outer = 0; j_outer < 31; j_outer += 1) {
      C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j_outer)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) + (16 * j_outer))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j_outer))));
    };
    for (int32_t j_outer = 31; j_outer < 32; j_outer += 1) {
      for (int32_t j_inner = 0; j_inner < (500 + (-16 * j_outer)); j_inner += 1) {
        C[((500 * i) + ((16 * j_outer) + j_inner))] = (A[((500 * i) + ((16 * j_outer) + j_inner))] * B[((500 * i) + ((16 * j_outer) + j_inner))]);
      };
    };
  };
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 500; j += 1) {
      D[((500 * i) + j)] = (A[((500 * i) + j)] * B[((500 * i) + j)]);
    };
  };
}
)ROC";

  // EXPECT_EQ(Trim(out), Trim(target_out));
  EXPECT_EQ(Context::Global().info_rgt().Get<int>("vectorized_forloop_count"), 1);
}

TEST(Vectorize, vectorize) {
  Var a("a");
  Var b("b");
  Var c("c");

  {
    Expr d = a * 10 + b;
    detail::Vectorize(a, 16, &d);
    Simplify(&d);
    EXPECT_EQ(GetStreamCnt(d), "Ramp(b,10,16)");
  }

  {
    Expr d = a * 10 + b;
    detail::Vectorize(b, 16, &d);
    Simplify(&d);
    EXPECT_EQ(GetStreamCnt(d), "Ramp((10 * a),1,16)");
  }

  {
    Placeholder<float> A("A", std::vector<int>{{10}});
    Placeholder<float> B("B", std::vector<int>{{10}});
    Placeholder<float> C("C", std::vector<int>{{10}});

    auto expr = Load::Make(ir::Tensor(A), {a * 2 + b * 2});
    expr      = expr + 10.f * expr;
    detail::Vectorize(a, 16, &expr);
    EXPECT_EQ(
        GetStreamCnt(expr),
        "(A[Ramp(((b * 2) + (0 * 2)),(1 * 2),16)] + (Broadcast(10,16) * A[Ramp(((b * 2) + (0 * 2)),(1 * 2),16)]))");
  }
}

TEST(Vectorize, single_for) {
  Placeholder<float> A("A", std::vector<int>{{10}});
  Placeholder<float> B("B", std::vector<int>{{10}});
  Placeholder<float> C("C", std::vector<int>{{10}});

  Var loop_var("k");

  Expr body = Store::Make(ir::Tensor(C),
                          ir::Add::Make(  //
                              ir::Load::Make(ir::Tensor(A), {Expr(loop_var)}),
                              ir::Load::Make(ir::Tensor(B), {Expr(loop_var)})),
                          {Expr(loop_var)});
  body      = ir::Block::Make({body});

  VectorizeInfo vectorize_info(0, 16);
  auto forloop = ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(16),
                               ir::ForType::Vectorized,
                               ir::DeviceAPI::UNK,
                               body,
                               vectorize_info);

  forloop = optim::Optimize(forloop);

  LOG(INFO) << "Forloop\n" << forloop;
}

}  // namespace optim
}  // namespace cinn
