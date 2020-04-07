#pragma once

#include <gflags/gflags.h>

#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/function.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/module.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {

//! Root of the builtin code.
DECLARE_string(cinn_x86_builtin_code_root);

namespace lang {
class Module;
}  // namespace lang

namespace backends {

class CodeGenC : public ir::IrPrinter {
 public:
  enum class OutputKind {
    CHeader,  //! output the C header file.
    CImpl,    //! output the C implementation file.
  };

  CodeGenC(Target target);

  void Compile(const lang::Module& module, const Outputs& outputs);

  std::string Compile(const lang::Module& module, OutputKind output_kind);

  //! Disable inline the builtin codes(too large) for simpler string comparation.
  void SetInlineBuiltinCodes(bool x = true) { inline_builtin_codes_ = x; }

 protected:
  std::string Compile(const ir::LoweredFunc& function);
  std::string Compile(const ir::Buffer& buffer);

  void GenerateHeaderFile(const lang::Module& module);

  std::string PrintType(Type type);
  //! type cast, print like "int(x)"
  // @{
  void PrintCastExpr(const Type& type, Expr e);
  void PrintCastExpr(const std::string& type, Expr e);
  // @}

  void PrintFunctionDeclaration(const ir::_LoweredFunc_* op) {
    os() << "void " << op->name << "(";
    os() << "void* _args, int32_t num_args";
    os() << ")";
  }

  void PrintShape(const std::vector<Expr>& shape);

  void PrintIncludes();
  void PrintBuiltinCodes();
  void PrintFileGuardOpen(const std::string& module_name);
  void PrintFileGuardClose(const std::string& module_name);

  //! Create the buffers in global scope(just creation without allocating them).
  void PrintBufferCreation(const std::vector<ir::Buffer>& buffers);
  void PrintBufferDestroy(const std::vector<ir::Buffer>& buffers);
  void PrintRuntimeType(const cinn_type_t& type);

#define __DEFINE_VISIT(op__) void Visit(const ir::op__* op) override;
  NODETY_FORALL(__DEFINE_VISIT)
#undef __DEFINE_VISIT

  void PrintFuncArg(const ir::Argument& arg);

  void PrintStackVecType(Type type, int lanes);

 protected:
  Target target_;
  std::stringstream ss_;
  bool inline_builtin_codes_{true};
};

namespace detail {

Expr StridedRampBase(Expr e, int stride);

}  // namespace detail

}  // namespace backends
}  // namespace cinn
