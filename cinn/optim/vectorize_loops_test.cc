#include "cinn/optim/vectorize_loops.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_simplify.h"
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
    C->stage()->Split(j_inner, 8, poly::SplitRestStrategy::kAuto);
  }

  // Code gen
  auto funcs = Lower("matmul", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  optim::VectorizeLoops(&funcs[0]->body, target);

  lang::Module module("module1", target);
  module.Append(funcs.front());
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
  {
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
  };
}
)ROC";

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

  optim::TransformPolyForToFor(&funcs[0]->body);

  detail::Vectorize(ir::_Var_::Make("j_inner", Int(32)), 16, &funcs.front()->body);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  lang::Module module("module1", target);
  module.Append(funcs[0]);

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
    for (int32_t j_outer = 0; j_outer < 31; j_outer += 1) {
      for (int32_t j_inner = 0; j_inner < 16; j_inner += 1) {
        C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j_outer)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) + (16 * j_outer))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j_outer))));
      };
    };
    for (int32_t j_outer = 31; j_outer < 32; j_outer += 1) {
      for (int32_t j_inner = 0; j_inner < (500 + (-16 * j_outer)); j_inner += 1) {
        C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j_outer)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) + (16 * j_outer))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j_outer))));
      };
    };
  };
}
)ROC";
  EXPECT_EQ(Trim(out), Trim(target_out));
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

  EXPECT_EQ(Trim(out), Trim(target_out));
  EXPECT_EQ(Context::Global().info_rgt().Get<int>("vectorized_forloop_count"), 1);
}

}  // namespace optim
}  // namespace cinn
