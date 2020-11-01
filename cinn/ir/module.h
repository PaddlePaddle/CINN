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

namespace ir {

/**
 * Module represents IR containing lowered function definitions and buffers.
 */
class Module : public ir::IrNodeRef {
 public:
  struct Builder {
    Builder(const std::string& name, const Target& target) : module_(common::make_shared<ir::_Module_>()) {
      module_->name   = name;
      module_->target = target;
    }

    void AddFunction(ir::LoweredFunc func);
    void AddBuffer(ir::Buffer buffer);

    Module Build();

   private:
    Shared<ir::_Module_> module_;
  };

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

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

  ir::_Module_* self();
  const ir::_Module_* self() const;

  ir::_Module_* operator->() { return self(); }
  const ir::_Module_* operator->() const { return self(); }

  operator Expr() const;

 protected:
  Module(const std::string& name, const Target& target);

  explicit Module(ir::IrNode* n) : ir::IrNodeRef(n) {}

  friend class Module::Builder;
  friend class backends::CodeGenC;
  friend class ::cinn::ir::Expr;
  friend class ::cinn::ir::_Module_;
};

}  // namespace ir
}  // namespace cinn
