#pragma once

#include <string>

#include "cinn/hlir/instruction/graph.h"
#include "cinn/hlir/instruction/pass.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace pass {

/**
 * This pass helps determine the `inline` or not for each instruction.
 */
class BufferAssignPass : public PassInterface {
 public:
  explicit BufferAssignPass(const std::string &name) : name_(name) {}
  std::string_view name() const override;

  bool Run(Module *module) override;

  bool RunOnModuleGroup(ModuleGroup *module_group) override;

  bool is_pass_pipeline() const override;

 protected:
  void RunOnComputation(Computation *comp);

  bool ShouldInline(Instruction *instr, Computation *comp) const;

 private:
  std::unordered_set<InstrCode> unary_codes_{{
      InstrCode::Tanh,
      InstrCode::Sigmoid,
      InstrCode::Ceil,
      InstrCode::Floor,
      InstrCode::Abs,
      InstrCode::Not,
  }};
  std::string name_;
};

}  // namespace pass
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
