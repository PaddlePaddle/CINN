#pragma once
#include "cinn/backends/codegen_c.h"
#include "cinn/common/common.h"
#include "cinn/ir/function.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/module.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {

namespace lang {
class Module;
}  // namespace lang

namespace backends {

/**
 * CUDA device code generator.
 */
class CodeGenCUDA_Dev : public CodeGenC {
 public:
  explicit CodeGenCUDA_Dev(Target target);

  /**
   * Compile the \p module to \p outputs.
   */
  void Compile(const lang::Module& module, const Outputs& outputs);

  std::string Compile(const ir::LoweredFunc& func);

 protected:
  void Visit(const ir::_LoweredFunc_* op) override;

 private:
  Target target_;
};

}  // namespace backends
}  // namespace cinn
