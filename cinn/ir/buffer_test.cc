#include "cinn/ir/buffer.h"

#include <gtest/gtest.h>

#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/common/common.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/module.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace ir {

TEST(Buffer, basic) {
  Var ptr("buff", Float(32));
  std::vector<Expr> shape({Expr(100), Expr(20)});
  Var i("i"), j("j");
  std::vector<Expr> strides({Expr(0), Expr(0)});
  auto buffer = _Buffer_::Make(ptr, ptr->type(), shape, strides, Expr(0), "buf", "", 0, 0);

  // Check shared
  ASSERT_EQ(ref_count(buffer.get()).val(), 1);

  {
    auto buffer1 = buffer;
    ASSERT_EQ(ref_count(buffer.get()).val(), 2);
    ASSERT_EQ(ref_count(buffer1.get()).val(), 2);
  }

  ASSERT_EQ(ref_count(buffer.get()).val(), 1);
}

TEST(Buffer, bind_to_multiple_tensors) {
  Tensor A = lang::Compute(
      {100, 20}, [=](Var i, Var j) { return Expr(0.f); }, "A");
  Tensor B = lang::Compute(
      {100, 20}, [=](Var i, Var j) { return Expr(1.f); }, "B");
  lang::Buffer buf0(A->type(), "buf0");

  A->Bind(buf0);
  B->Bind(buf0);

  auto funcs = lang::Lower("func1", {A, B});

  ASSERT_EQ(funcs.size(), 1UL);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os   = Target::OS ::Linux;

  lang::Module module("module1", target);
  module.Append(funcs.front());
  module.Append(buf0);

  backends::CodeGenC codegen(target);
  auto out = codegen.Compile(module, backends::CodeGenC::OutputKind::CImpl);
  std::cout << "codegen C:" << std::endl << out << std::endl;
}

}  // namespace ir
}  // namespace cinn
