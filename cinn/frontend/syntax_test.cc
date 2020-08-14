#include "cinn/frontend/syntax.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn {
namespace frontend {

TEST(syntax, basic) {
  const int M = 32;
  const int N = 24;

  Placeholder a(Float(32), {M, N});
  Placeholder b(Float(32), {M, N});
  Program program;

  auto c = program.add(a, b);
  auto d = program.add(a, c);

  // output program
  for (int i = 0; i < program.size(); i++) {
    LOG(INFO) << "instruction: " << program[i];
  }
}

}  // namespace frontend
}  // namespace cinn
