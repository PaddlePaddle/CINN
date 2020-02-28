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
 * Content of a module.
 */
struct _Module_ : Object {
  std::string name;
  Target target;
  std::vector<ir::Buffer> buffers;
  std::vector<ir::LoweredFunc> functions;
  std::vector<Module> submodules;

  const char* type_info() const override { return "_Module_"; }
};

/**
 * Module represents IR containing lowered function definitions and buffers.
 */
class Module {
 public:
  Module(const std::string& name, const Target& target);

  //! Get the target of this module.
  const Target& target() const;

  //! Get the name of the module.
  const std::string& name() const;

  //! The members in the module.
  // @{
  const std::vector<ir::Buffer>& buffers() const;
  const std::vector<ir::LoweredFunc>& functions() const;
  const std::vector<Module>& submodules() const;
  // @}

  //! Add something to this module.
  // @{
  void Append(const Buffer& buffer);
  void Append(const ir::LoweredFunc& function);
  void Append(const Module& module);
  // @}

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

  _Module_* self();
  const _Module_* self() const;

  _Module_* operator->() { return self(); }
  const _Module_* operator->() const { return self(); }

 protected:
  std::vector<Expr> buffer_creation_exprs() const;

  friend class backends::CodeGenC;

 private:
  Shared<_Module_> module_;
};

}  // namespace lang
}  // namespace cinn
