#pragma once

#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/function.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/module.h"

namespace cinn {

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

  CodeGenC(std::ostream& os, Target target, OutputKind output_kind);

  void Compile(const lang::Module& module);
  void Compile(const ir::LoweredFunc& function);
  void Compile(const ir::Buffer& buffer);

 protected:
  void GenerateHeaderFile(const lang::Module& module);

  std::string PrintType(Type type);
  void PrintCastExpr(const Type& type, Expr e);

  void PrintIncludes();
  void PrintFileGuardOpen(const std::string& module_name);
  void PrintFileGuardClose(const std::string& module_name);
  //! Create the buffers in global scope(just creation without allocating them).
  void PrintBufferCreation(const std::vector<ir::Buffer>& buffers);
  void PrintBufferDestroy(const std::vector<ir::Buffer>& buffers);

#define __DEFINE_VISIT(op__) void Visit(const ir::op__* op) override;
  NODETY_FORALL(__DEFINE_VISIT)
#undef __DEFINE_VISIT

  void PrintFuncArg(const ir::Argument& arg) {
    if (arg.is_buffer()) {
      if (arg.is_input()) {
        os() << "const struct cinn_buffer_t *";
      } else {
        os() << "struct cinn_buffer_t *";
      }
    } else if (arg.is_scalar()) {
      os() << PrintType(arg.type) << " ";
      os() << arg.name;
    } else {
      NOT_IMPLEMENTED
    }
    os() << arg.name;
  }

 private:
  Target target_;
  OutputKind output_kind_;
};

}  // namespace backends
}  // namespace cinn
