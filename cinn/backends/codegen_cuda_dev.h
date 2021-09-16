#pragma once
#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/ir/module.h"
#include "cinn/lang/packed_func.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn::ir {
class Module;
}  // namespace cinn::ir

namespace cinn {
namespace backends {

/**
 * CUDA device code generator.
 *
 * It generates the device function, e.g, the function called "myadd" will have a __global__ functon called
 * "myadd_kernel", different from codegen_c, the declaration of the "myadd_kernel" function has an expanded argument
 * list, which finally similar to `__global__ void myadd(float* __restrict__ A, float* __restrict__ B, int n);`
 */
class CodeGenCUDA_Dev : public CodeGenC {
 public:
  explicit CodeGenCUDA_Dev(Target target);

  /**
   * Compile the \p module to \p outputs.
   */
  void Compile(const ir::Module& module, const Outputs& outputs);

  //! Compile on NVRTC.
  std::string Compile(const ir::Module& module, bool for_nvrtc = true);

  std::string Compile(const ir::LoweredFunc& func);

  /**
   * Generate the kernel function's name given a function.
   */
  static std::string GenKernelName(const std::string& func_name) { return func_name + "_kernel"; }

  /**
   * \brief Print a function argument in CUDA syntax. Currently, just some decoration of __restrict__.
   * @param arg the argument.
   * @return the representation in CUDA syntax.
   *
   * We make it a static to make the test easier.
   */
  void PrintFuncArg(const ir::Argument& arg);

  std::string Compile(const ir::Module& module, OutputKind output_kind);

 protected:
  void Visit(const ir::_LoweredFunc_* op) override;
  void Visit(const ir::Min* op) override;
  void Visit(const ir::Max* op) override;
  void Visit(const ir::Alloc* op) override;
  void Visit(const ir::Call* op) override;

  void PrintBuiltinCodes();

  void PrintIncludes() override;

  void PrintTempBufferCreation(const ir::Buffer& buffer);

  void PrintTempBufferAliasDefinition(const ir::Buffer& buffer);

  std::vector<Expr> GenerateBufferAliasExprs(const ir::_LoweredFunc_* op, const std::vector<ir::Buffer>& temp_buffers);

  /**
   * Print the function declaration, this is different from C, we expand the arguments and get something like
   * `__global__ void myadd(float* __restrict__ A, float* __restrict__ B, int n);`
   */
  void PrintFunctionDeclaration(const ir::_LoweredFunc_* op);

 private:
  Target target_;
  bool for_nvrtc_{false};
};

}  // namespace backends
}  // namespace cinn
