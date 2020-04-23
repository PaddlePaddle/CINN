#include "hlir/instruction/lower.h"
#include <gtest/gtest.h>
#include <utility>
#include "cinn/backends/codegen_c.h"

namespace hlir {
namespace instruction {
using cinn::Expr;

std::unique_ptr<Computation> create_elementwise_add(Context* context, cinn::Var N, const std::string& name) {
  Computation::Builder builder(context, name);
  {
    ParameterConfig parameter_config = {Float(32)};
    auto x  = builder.AddInstruction(Instruction::CreateParameter(0, Shape({N, 30, 40}), "x", parameter_config));
    auto w  = builder.AddInstruction(Instruction::CreateParameter(1, Shape({N, 30, 40}), "w", parameter_config));
    auto w1 = builder.AddInstruction(Instruction::CreateParameter(1, Shape({N, 30, 40}), "w1", parameter_config));

    auto add  = builder.AddInstruction(Instruction::CreateBinary(x->shape(), InstrCode::Add, x, w));
    auto add1 = builder.AddInstruction(Instruction::CreateBinary(x->shape(), InstrCode::Add, add, w1));

    add->set_inlined();
  }

  return builder.Build();
}

TEST(Lower, computation) {
  Context context;
  cinn::Var N("N");

  auto comp0 = create_elementwise_add(&context, N, "elementwise_add");

  LOG(INFO) << "HLIR:\n" << comp0->to_debug_string();

  ComputationLower lower(nullptr);
  Expr fn = lower(comp0.get());

  LOG(INFO) << "\n" << fn;
}

TEST(Lower, module) {
  Context context;
  cinn::Var N("N");

  auto comp0 = create_elementwise_add(&context, N, "elementwise_add");
  auto comp1 = create_elementwise_add(&context, N, "elementwise_add1");
  auto comp2 = create_elementwise_add(&context, N, "elementwise_add2");

  Module module("module1");
  module.AddComputation(std::move(comp0));
  module.AddComputation(std::move(comp1));
  module.AddEntryComputation(std::move(comp2));

  LOG(INFO) << "HLIR:\n" << module.to_debug_string();

  ModuleLower lower;
  auto cinn_module = lower(&module);

  cinn::backends::CodeGenC codegen{cinn::Target()};
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(cinn_module, cinn::backends::CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "C code: \n" << out;
}

}  // namespace instruction
}  // namespace hlir
