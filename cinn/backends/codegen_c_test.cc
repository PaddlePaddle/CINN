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

  lang::Buffer C_buf;
  auto C = lang::Compute(
      {100, 20}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  C->Bind(C_buf);
  return std::make_tuple(A, B, C, C_buf);
}

TEST(CodeGenC, basic) {
  std::stringstream ss;
  Target target;
  CodeGenC codegen(ss, target);

  ir::Tensor A, B, C;
  lang::Buffer C_buf;
  std::tie(A, B, C, C_buf) = CreateTensor1();

  auto funcs = lang::Lower("func_C", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  codegen.Compile(funcs.front());

  auto out = ss.str();

  std::cout << "codegen C:" << std::endl << out << std::endl;

  EXPECT_EQ(utils::Trim(out),
            utils::Trim(
                R"ROC(
void func_C(struct cinn_buffer_t *A, struct cinn_buffer_t *B, struct cinn_buffer_t *C)
{
  for (int32_t c1 = 0; (c1 <= 99); c1 += 1){
    for (int32_t c3 = 0; (c3 <= 19); c3 += 1){
      C[((c1 * 20) + c3)] = (A(c1, c3) + B(c1, c3));
    };
  };
}
)ROC"));
}

TEST(CodeGenC, module) {
  ir::Tensor A, B, C;
  lang::Buffer C_buf;
  std::tie(A, B, C, C_buf) = CreateTensor1();

  Target target;
  lang::Module module("module1", target);

  auto funcs = lang::Lower("add1", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  module.Append(funcs.front());
  module.Append(C_buf);

  std::stringstream ss;
  CodeGenC codegen(ss, target);
  codegen.Compile(module);

  auto out = ss.str();
  std::cout << "codegen C:" << std::endl << out << std::endl;
}

}  // namespace backends
}  // namespace cinn
