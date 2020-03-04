#include "cinn/backends/codegen_c.h"

#include <gtest/gtest.h>

#include <sstream>

#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace backends {

std::tuple<ir::Tensor, ir::Tensor, ir::Tensor, lang::Buffer> CreateTensor1() {
  lang::Placeholder<float> A("A", {100, 20});
  lang::Placeholder<float> B("B", {100, 20});

  lang::Buffer C_buf(Float(32));
  auto C = lang::Compute(
      {100, 20}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);
  return std::make_tuple(A, B, C, C_buf);
}

TEST(CodeGenC, module) {
  ir::Tensor A, B, C;
  lang::Buffer C_buf(Float(32));
  std::tie(A, B, C, C_buf) = CreateTensor1();

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  lang::Module module("module1", target);

  auto funcs = lang::Lower("add1", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  module.Append(funcs.front());
  module.Append(C_buf);

  {
    std::stringstream ss;
    CodeGenC codegen(target);
    auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
    std::cout << "codegen C:" << std::endl << out << std::endl;

    std::string target_str = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t());
void add1(struct cinn_buffer_t *_A, struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _A);
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), _C);
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  float* B = (float*)(cinn_buffer_get_data_handle(_B));
  float* A = (float*)(cinn_buffer_get_data_handle(_A));
  for (int32_t i = 0; (i <= 99); i += 1){
    for (int32_t j = 0; (j <= 19); j += 1){
      C[((i * 20) + j)] = (A[((i * 20) + j)] + B[((i * 20) + j)]);
    };
  };
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

void add1(struct cinn_buffer_t *_A, struct cinn_buffer_t *_B, struct cinn_buffer_t *_C);


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
  lang::Placeholder<float> A("A", {100, 20});
  lang::Placeholder<float> B("B", {100, 20});

  lang::Buffer C_buf(Float(32)), D_buf(Float(32));

  // An inlined tensor, should not appear in final C code! It can be used by any times and expand its expression there.
  auto inlined0 = lang::Compute({100, 20}, [&](Var i, Var j) { return A(i, j) * 2.f + 1.f; });

  auto C = lang::Compute(
      {100, 20}, [&](Var i, Var j) { return A(i, j) + B(i, j) + inlined0(i, j); }, "C");
  C->Bind(C_buf);

  auto D = lang::Compute(
      {100, 20}, [&](Var i, Var j) { return C(i, j) * 2.f * inlined0(i, j); }, "D");
  D->Bind(D_buf);

  poly::Iterator i_outer, i_inner;
  std::tie(i_outer, i_inner) = C->stage()->Split(poly::DefaultIterator(0), 4);

  D->stage()->Tile(poly::DefaultIterator(0), poly::DefaultIterator(1), 4, 16);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;
  lang::Module module("module1", target);

  auto funcs = lang::Lower("add1", {A, B, C, D});

  ASSERT_EQ(funcs.size(), 1UL);

  module.Append(funcs.front());
  module.Append(C_buf);

  CodeGenC codegen(target);
  auto out = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;

  auto tgt = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

cinn_buffer_t* C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t());
void add1(struct cinn_buffer_t *_A, struct cinn_buffer_t *_B, struct cinn_buffer_t *_C, struct cinn_buffer_t *_D)
{
  cinn_buffer_malloc((void*)(0), _A);
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _D);
  float* D = (float*)(cinn_buffer_get_data_handle(_D));
  float* C = (float*)(cinn_buffer_get_data_handle(_C));
  float* A = (float*)(cinn_buffer_get_data_handle(_A));
  float* B = (float*)(cinn_buffer_get_data_handle(_B));
  for (int32_t i_outer = 0; (i_outer <= 24); i_outer += 1){
    for (int32_t i_inner = 0; (i_inner <= 3); i_inner += 1){
      for (int32_t j = 0; (j <= 19); j += 1){
        C[((((4 * i_outer) + i_inner) * 20) + j)] = ((A[((((4 * i_outer) + i_inner) * 20) + j)] + B[((((4 * i_outer) + i_inner) * 20) + j)]) + ((A[((((4 * i_outer) + i_inner) * 20) + j)] * 2) + 1));
      };
    };
  };
  for (int32_t i_outer = 0; (i_outer <= 24); i_outer += 1){
    for (int32_t i_inner = 0; (i_inner <= 3); i_inner += 1){
      for (int32_t j_outer = 0; (j_outer <= 1); j_outer += 1){
        for (int32_t j_inner = 0; (j_inner <= min(15, ((-16 * j_outer) + 19))); j_inner += 1){
          D[((((4 * i_outer) + i_inner) * 20) + ((16 * j_outer) + j_inner))] = ((C[((((4 * i_outer) + i_inner) * 20) + ((16 * j_outer) + j_inner))] * 2) * ((A[((((4 * i_outer) + i_inner) * 20) + ((16 * j_outer) + j_inner))] * 2) + 1));
        };
      };
    };
  };
}
)ROC";

  ASSERT_EQ(utils::Trim(tgt), utils::Trim(out));
}

}  // namespace backends
}  // namespace cinn
