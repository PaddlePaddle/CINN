#include "hlir/instruction/computation.h"
#include <gtest/gtest.h>
#include <iostream>

namespace hlir {
namespace instruction {

TEST(Computation, basic) {
  Context context;
  Computation::Builder builder(&context, "default_module");
  ParameterConfig parameter_config = {Float(32)};
  auto x                           = builder.AddInstruction(
      Instruction::CreateParameter(0, Shape({Shape::kDynamicDimValue, 30, 40}), "x", parameter_config));
  auto w = builder.AddInstruction(Instruction::CreateParameter(1, Shape({40, 50}), "w", parameter_config));

  auto dot0     = builder.AddInstruction(Instruction::CreateDot(Shape({30, 50}), x, w), "DOT");
  auto constant = builder.AddInstruction(Instruction::CreateConstant(Shape({1}), {1}, {Float(32)}), "constant 1");
  auto add      = builder.AddInstruction(Instruction::CreateBinary(Shape({30, 50}), InstrCode::Add, dot0, constant));

  auto computation = builder.Build();

  std::cout << "computation:\n" << computation->to_debug_string() << std::endl;
}

}  // namespace instruction
}  // namespace hlir
