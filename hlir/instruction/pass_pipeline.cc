#include "hlir/instruction/pass_pipeline.h"

namespace hlir {
namespace instruction {

bool PassPipeline::Run(Module* module) {
  bool changed = false;
  for (auto& pass : passes_) {
    changed |= pass->Run(module);
  }
  return changed;
}

bool PassPipeline::RunOnModuleGroup(ModuleGroup* module_group) {
  bool changed = false;
  for (auto& pass : passes_) {
    changed |= pass->RunOnModuleGroup(module_group);
  }
  return changed;
}

}  // namespace instruction
}  // namespace hlir
