#pragma once

#include <memory>

#include "cinn/backends/llvm/simple_orc_jit.h"
#include "hlir/instruction/instruction.h"
#include "hlir/instruction/module.h"

namespace hlir {
namespace instruction {

using lowered_func_p = void (*)(void*, int32_t);

class Compiler {
 public:
  Compiler();

  /**
   * Evaluate a module.
   */
  void Eval(const Module* module, cinn_pod_value_t* args, int args_num, const std::string& fn_name = "");

  void Eval(const std::string& name, cinn_pod_value_t* args, int args_num);

  /**
   * Compile the module.
   * @param module
   */
  lowered_func_p Compile(const Module* module);

  lowered_func_p Lookup(const std::string& name) const;

 private:
  std::unique_ptr<cinn::backends::SimpleOrcJit> jit_;
};

}  // namespace instruction
}  // namespace hlir
