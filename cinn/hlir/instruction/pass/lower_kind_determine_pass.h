#pragma once

#include <string>

#include "cinn/hlir/instruction/graph.h"
#include "cinn/hlir/instruction/pass.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace pass {

/**
 * This pass set the default lower kind of all the instructions.
 */
class LowerKindDetermine : public PassInterface {
 public:
  explicit LowerKindDetermine(const std::string &name) : name_(name) {}
  std::string_view name() const override;

  bool Run(Module *module) override;

  bool RunOnModuleGroup(ModuleGroup *module_group) override;

  bool is_pass_pipeline() const override;

 protected:
  void RunOnComputation(Computation *comp);

  bool ShouldInline(Instruction *instr, Computation *comp) const;

 private:
  std::string name_;
};

}  // namespace pass
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
