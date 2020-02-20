#include "cinn/lang/module.h"

namespace cinn {
namespace lang {

/**
 * Content of a module.
 */
struct _Module_ : Object {
  std::string name;
  Target target;
  std::vector<ir::Buffer> buffers;
  std::vector<ir::PackedFunc> functions;
  std::vector<Module> submodules;

  const char *type_info() const override { return "_Module_"; }
};

_Module_ *Module::self() { return module_->As<_Module_>(); }
const _Module_ *Module::self() const { return module_->As<_Module_>(); }

Module::Module(const std::string &name, const Target &target) : module_(make_shared<_Module_>()) {
  self()->name   = name;
  self()->target = target;
}

const Target &Module::target() const { return self()->target; }

const std::string &Module::name() const { return self()->name; }

const std::vector<ir::Buffer> &Module::buffers() const { return self()->buffers; }

const std::vector<ir::PackedFunc> &Module::functions() const { return self()->functions; }

const std::vector<Module> &Module::submodules() const { return self()->submodules; }

void Module::Append(const ir::Buffer &buffer) { self()->buffers.push_back(buffer); }

void Module::Append(const ir::PackedFunc &function) { self()->functions.push_back(function); }

void Module::Append(const Module &module) { self()->submodules.push_back(module); }

void Module::Compile(const backends::Outputs &outputs) const {}

}  // namespace lang
}  // namespace cinn
