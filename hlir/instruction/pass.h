#pragma once

#include "hlir/instruction/module.h"

namespace hlir {
namespace instruction {

struct ModuleGroup;

/**
 * The base class of all Passes.
 */
class PassInterface {
 public:
  virtual ~PassInterface()              = default;
  virtual std::string_view name() const = 0;

  virtual bool Run(Module* module)                         = 0;
  virtual bool RunOnModuleGroup(ModuleGroup* module_group) = 0;

  virtual bool is_pass_pipeline() const { return false; }
};

/**
 * The base class of the Passes performed on Modules.
 */
class ModulePass : public PassInterface {
 public:
  bool RunOnModuleGroup(ModuleGroup* module_group) override;
};

/**
 * The base class of the Passes performed on ModuleGroups.
 */
class ModuleGroupPass : public PassInterface {
 public:
  bool Run(Module* module) override { NOT_IMPLEMENTED }
};

}  // namespace instruction
}  // namespace hlir
