#include "hlir/instruction/compiler.h"

#include "cinn/backends/llvm/simple_orc_jit.h"
#include "cinn/runtime/cinn_runtime.h"
#include "hlir/instruction/lower.h"

namespace hlir {
namespace instruction {

Compiler::Compiler() { jit_ = cinn::backends::SimpleOrcJit::Create(); }

void Compiler::Eval(const std::string &name, cinn_pod_value_t *args, int args_num) {
  auto addr = jit_->Lookup(name);
  auto *fn  = reinterpret_cast<void (*)(void *, int32_t)>(addr);
  CHECK(fn);

  fn(args, args_num);
}

void Compiler::Eval(const Module *module, cinn_pod_value_t *args, int args_num, const std::string &fn_name) {
  // TODO(Superjomn) make a cache here.

  lowered_func_p fn{};

  auto main_fn = Compile(module);

  if (fn_name.empty()) {
    fn = main_fn;
  } else {
    auto addr = jit_->Lookup(fn_name);
    fn        = reinterpret_cast<void (*)(void *, int32_t)>(addr);
  }

  CHECK(fn);
  fn(args, args_num);
}

lowered_func_p Compiler::Compile(const Module *module) {
  auto cinn_module = Lower(*module, true);

  jit_->Link(cinn_module, /*optimize=*/false);

  if (module->entry_computation()) {
    auto entry_fn_name = module->entry_computation()->name();

    auto main_fn_addr = jit_->Lookup(entry_fn_name);

    auto main_fn = reinterpret_cast<void (*)(void *, int32_t)>(main_fn_addr);
    return main_fn;
  } else {
    return nullptr;
  }
}
lowered_func_p Compiler::Lookup(const std::string &name) const {
  return reinterpret_cast<lowered_func_p>(jit_->Lookup(name));
}

}  // namespace instruction
}  // namespace hlir
