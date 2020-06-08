#include "cinn/hlir/instruction/pass/lower_kind_determine_pass.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace pass {

std::string_view LowerKindDetermine::name() const { return name_; }

bool LowerKindDetermine::Run(Module *module) {
  for (auto &item : module->computations()) {
    RunOnComputation(item.second.get());
  }

  return true;
}

bool LowerKindDetermine::RunOnModuleGroup(ModuleGroup *module_group) { return false; }
bool LowerKindDetermine::is_pass_pipeline() const { return PassInterface::is_pass_pipeline(); }

void LowerKindDetermine::RunOnComputation(Computation *comp) {
  for (auto &instr : comp->instructions()) {
    if (instr->lower_kind() == "none") {
      instr->set_lower_kind("base");
    }
  }
}

bool LowerKindDetermine::ShouldInline(Instruction *instr, Computation *comp) const { return false; }

}  // namespace pass
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_PASS(lower_kind_determine, LowerKindDetermine)
