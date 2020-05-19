#include "cinn/hlir/instruction/module.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace instruction {

TEST(Module, basic) {
  Module module("module1");

  cinn::Var N("N");
  Context context;
  Computation::Builder builder(&context, "func0");

  ParameterConfig parameter_config = {Float(32)};
  auto x = builder.AddInstruction(Instruction::CreateParameter(0, Shape({N, 30, 40}), "x", parameter_config));
  auto w = builder.AddInstruction(Instruction::CreateParameter(1, Shape({40, 50}), "w", parameter_config));

  auto dot0     = builder.AddInstruction(Instruction::CreateDot(x, w, Shape({30, 50})), "DOT");
  auto constant = builder.AddInstruction(Instruction::CreateConstant(Shape({1}), {1}, {Float(32)}), "constant 1");
  auto add      = builder.AddInstruction(Instruction::CreateBinary(InstrCode::Add, dot0, constant, Shape({30, 50})));

  auto* comp0 = module.AddComputation(builder.Build());

  // call
  Computation::Builder builder1(&context, "main");
  auto arg0   = builder1.AddInstruction(Instruction::CreateParameter(0, Shape({N, 30, 40}), "x", parameter_config));
  auto arg1   = builder1.AddInstruction(Instruction::CreateParameter(1, Shape({40, 50}), "w", parameter_config));
  auto* call0 = builder1.AddInstruction(
      Instruction::CreateCall({arg0, arg1}, "call0_ret", Shape({N, 30, 50}), Float(32), comp0), "call");
  auto* call1 = builder1.AddInstruction(Instruction::CreateCustomCall({}, {arg0, arg1}, "mkl_gemm", ""));
  auto* comp1 = module.AddEntryComputation(builder1.Build());

  std::cout << module.to_debug_string() << std::endl;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
