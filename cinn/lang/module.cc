#include "cinn/lang/module.h"

namespace cinn {
namespace lang {

_Module_ *Module::self() { return p_->as<_Module_>(); }
const _Module_ *Module::self() const { return p_->as<_Module_>(); }

Module::Module(const std::string &name, const Target &target) : Shared<_Module_>(make_shared<_Module_>()) {
  self()->name   = name;
  self()->target = target;
}

const Target &Module::target() const { return self()->target; }

const std::string &Module::name() const { return self()->name; }

const std::vector<ir::Buffer> &Module::buffers() const { return self()->buffers; }

const std::vector<ir::LoweredFunc> &Module::functions() const { return self()->functions; }

const std::vector<Module> &Module::submodules() const { return self()->submodules; }

void Module::Append(const Buffer &buffer) {
  self()->buffers.push_back(buffer.buffer());
  self()->buffers.back()->target = target();
}

void Module::Append(const ir::LoweredFunc &function) { self()->functions.push_back(function); }

void Module::Append(const Module &module) { self()->submodules.push_back(module); }

void Module::Compile(const backends::Outputs &outputs) const {}

}  // namespace lang
}  // namespace cinn
