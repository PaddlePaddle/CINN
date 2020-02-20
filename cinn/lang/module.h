#pragma once
#include <string>
#include <vector>

#include "cinn/backends/outputs.h"
#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/function.h"

namespace cinn {
namespace lang {

class _Module_;

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
  const std::vector<ir::PackedFunc>& functions() const;
  const std::vector<Module>& submodules() const;
  // @}

  //! Add something to this module.
  // @{
  void Append(const ir::Buffer& buffer);
  void Append(const ir::PackedFunc& function);
  void Append(const Module& module);
  // @}

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

 private:
  Shared<_Module_> module_;
};

}  // namespace lang
}  // namespace cinn
