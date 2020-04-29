#include "hlir/instruction/compiler.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "hlir/instruction/module.h"

namespace hlir {
namespace instruction {

auto CreateTestBuffer(int kM, int kN) {
  auto* A = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto* B = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  auto* C = cinn_buffer_t::new_(cinn_device_kind_t::cinn_x86_device, cinn_float32_t(), {kM, kN}, 32);
  cinn_buffer_malloc(nullptr, A);
  cinn_buffer_malloc(nullptr, B);
  cinn_buffer_malloc(nullptr, C);
  float* Ad = reinterpret_cast<float*>(A->host_memory);
  float* Bd = reinterpret_cast<float*>(B->host_memory);

  for (int i = 0; i < A->num_elements(); i++) {
    Ad[i] = i;
    Bd[i] = i;
  }

  float* Cd = reinterpret_cast<float*>(C->host_memory);
  CHECK_EQ(C->num_elements(), A->num_elements());

  return std::make_tuple(A, B, C);
}

std::unique_ptr<Module> CreateModule(const std::string& name) {
  std::unique_ptr<Module> module(new Module(name));
  Context context;

  const char* fn_name = "elementwise_add0";

  {  // computation 0
    Computation::Builder builder(&context, fn_name);
    auto* x   = builder.AddInstruction(Instruction::CreateParameter(0, Shape({100, 200}), "X", {Float(32)}));
    auto* y   = builder.AddInstruction(Instruction::CreateParameter(1, Shape({100, 200}), "Y", {Float(32)}));
    auto* out = builder.AddInstruction(Instruction::CreateBinary(Shape({100, 200}), InstrCode::Add, x, y));
    out->set_inlined(false);  // the output should not be inlined

    module->AddComputation(builder.Build());
  }

  {  // computation main
    auto* target_computation = module->LookupComputation(fn_name);
    CHECK(target_computation);

    Computation::Builder builder(&context, "main");
    auto* x     = builder.AddInstruction(Instruction::CreateParameter(0, Shape({100, 200}), "X", {Float(32)}));
    auto* y     = builder.AddInstruction(Instruction::CreateParameter(1, Shape({100, 200}), "Y", {Float(32)}));
    auto* call0 = builder.AddInstruction(
        Instruction::CreateCall({x, y}, "out", Shape({100, 200}), Float(32), target_computation));
    auto* out_tuple = builder.AddInstruction(Instruction::CreateTuple(call0));
    auto* out       = builder.AddInstruction(Instruction::CreateTupleGet(out_tuple, 0));

    module->AddEntryComputation(builder.Build());
  }

  return module;
}

TEST(Compiler, basic) {
  Compiler compiler;

  auto module = CreateModule("module0");

  auto [a, b, c] = CreateTestBuffer(100, 200);  // NOLINT

  std::vector<cinn_pod_value_t> args({
      cinn_pod_value_t(a),
      cinn_pod_value_t(b),
      cinn_pod_value_t(c),
  });

  cinn_pod_value_to_buffer_p(args[0]);
  cinn_pod_value_to_buffer_p(args[1]);
  cinn_pod_value_to_buffer_p(args[2]);

  // TODO(fc) fix this
  // compiler.Eval(module.get(), args.data(), args.size());
}

}  // namespace instruction
}  // namespace hlir
