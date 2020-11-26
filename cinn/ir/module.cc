#include "cinn/ir/module.h"

#include <memory>

#include "cinn/optim/optimize.h"

namespace cinn {
namespace ir {

void Module::Builder::AddFunction(ir::LoweredFunc func) { module_->functions.push_back(func); }

void Module::Builder::AddBuffer(ir::Buffer buffer) {
  CHECK(buffer->target.defined()) << "buffer [" << buffer->name << "]'s target is undefined";
  if (std::find_if(module_->buffers.begin(), module_->buffers.end(), [&](const Expr &x) {
        return x.as_buffer()->name == buffer->name;
      }) == std::end(module_->buffers)) {
    module_->buffers.push_back(buffer);
    if (module_->target.arch == Target::Arch::X86) {
      module_->buffers.back().as_buffer()->data_alignment = 32;
    }
  }
}

Module Module::Builder::Build() {
  if (module_->functions.empty()) {
    LOG(ERROR) << "Module has no functions";
  }

  auto res = ir::Module(module_.get());

  return optim::Optimize(res, module_->target);
}

ir::_Module_ *Module::self() { return p_->as<ir::_Module_>(); }
const ir::_Module_ *Module::self() const { return p_->as<ir::_Module_>(); }

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
  std::vector<ir::Module> modules;
  for (auto &x : self()->submodules) {
    modules.push_back(x.as_module_ref());
  }
  return modules;
}

void Module::Compile(const backends::Outputs &outputs) const {}

Module::operator Expr() const { return Expr(ptr()); }

}  // namespace ir
}  // namespace cinn
