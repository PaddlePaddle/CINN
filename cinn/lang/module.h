#pragma once
#include <string>
#include <vector>

#include "cinn/backends/outputs.h"
#include "cinn/common/common.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/buffer.h"

namespace cinn {

namespace backends {
class CodeGenC;
}  // namespace backends

namespace lang {

class Module;

/**
 * Module represents IR containing lowered function definitions and buffers.
 */
class Module : ir::IrNodeRef {
 public:
  explicit Module(ir::IrNode* n) : ir::IrNodeRef(n) {}
  Module(const std::string& name, const Target& target);

  //! Get the target of this module.
  const Target& target() const;

  //! Get the name of the module.
  const std::string& name() const;

  //! The members in the module.
  // @{
  std::vector<ir::Buffer> buffers() const;
  std::vector<ir::LoweredFunc> functions() const;
  std::vector<Module> submodules() const;
  // @}

  //! Add something to this module, once added to a module, the buffer, function's target will be set with the module's
  //! target.
  // @{
  void Append(const Buffer& buffer);
  void Append(const ir::LoweredFunc& function);
  void Append(const Module& module);
  // @}

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

  ir::_Module_* self();
  const ir::_Module_* self() const;

  ir::_Module_* operator->() { return self(); }
  const ir::_Module_* operator->() const { return self(); }

 protected:
  friend class backends::CodeGenC;
};

}  // namespace lang
}  // namespace cinn
