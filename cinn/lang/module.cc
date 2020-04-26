#include "cinn/lang/module.h"
#include <memory>

namespace cinn {
namespace lang {

ir::_Module_ *Module::self() { return p_->as<ir::_Module_>(); }
const ir::_Module_ *Module::self() const { return p_->as<ir::_Module_>(); }

Module::Module(const std::string &name, const Target &target) : IrNodeRef(make_shared<ir::_Module_>()) {
  self()->name   = name;
  self()->target = target;
}

const Target &Module::target() const { return self()->target; }

const std::string &Module::name() const { return self()->name; }

std::vector<ir::Buffer> Module::buffers() const {
  std::vector<ir::Buffer> buffers;
  for (auto &buffer : self()->buffers) {
    buffers.emplace_back(buffer.as_buffer_ref());
  }
  return buffers;
}

std::vector<ir::LoweredFunc> Module::functions() const {
  std::vector<ir::LoweredFunc> functions;
  for (auto &x : self()->functions) {
    functions.emplace_back(x.as_lowered_func_ref());
  }
  return functions;
}

std::vector<Module> Module::submodules() const {
  std::vector<lang::Module> modules;
  for (auto &x : self()->submodules) {
    modules.push_back(x.as_module_ref());
  }
  return modules;
}

void Module::Append(const Buffer &buffer) {
  CHECK(buffer->target.defined()) << "buffer [" << buffer->name << "] not set";
  self()->buffers.push_back(buffer.buffer());
}

void Module::Append(const ir::LoweredFunc &function) { self()->functions.push_back(function); }

void Module::Append(const Module &module) { self()->submodules.emplace_back(module.get()); }

void Module::Compile(const backends::Outputs &outputs) const {}

}  // namespace lang
}  // namespace cinn
