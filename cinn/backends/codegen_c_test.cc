#include "cinn/backends/codegen_c.h"

#include <gtest/gtest.h>

#include <sstream>

#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace backends {

TEST(CodeGenC, basic) {
  std::stringstream ss;
  Target target;
  CodeGenC codegen(ss, target);

  lang::Placeholder<float> A("A", {100, 20});
  lang::Placeholder<float> B("B", {100, 20});

  lang::Buffer C_buf;
  auto C = lang::Compute({100, 20}, [&](Var i, Var j) { return A(i, j) + B(i, j); });
  C->Bind(C_buf);

  auto funcs = lang::Lower("func_C", {A, B, C});
  ASSERT_EQ(funcs.size(), 1UL);

  codegen.Compile(funcs.front());

  auto out = ss.str();

  std::cout << "codegen C:" << std::endl << out << std::endl;
}

}  // namespace backends
}  // namespace cinn
