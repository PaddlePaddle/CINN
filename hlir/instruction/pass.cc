#include "hlir/instruction/pass.h"

#include "hlir/instruction/module_group.h"

namespace hlir {
namespace instruction {

bool ModulePass::RunOnModuleGroup(ModuleGroup* module_group) {
  bool changed = false;
  for (auto* module : *module_group) {
    changed |= Run(module);
  }
  return changed;
}

}  // namespace instruction
}  // namespace hlir
