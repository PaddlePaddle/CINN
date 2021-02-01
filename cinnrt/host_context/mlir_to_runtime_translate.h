#pragma once

#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <string>
#include <unordered_map>

namespace cinnrt::host_context {

class CoreRuntimeBuilder;
class Value;
class ValueRef;
class KernelRegistry;
class MlirFunction;

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

/**
 * MlirToRuntimeTranslator helpes to translate a MLIR to a CoreRuntime.
 */
class MlirToRuntimeTranslator {
 public:
  using function_table_t = std::unordered_map<std::string, std::unique_ptr<MlirFunction>>;

  MlirToRuntimeTranslator(CoreRuntimeBuilder* runtime);
  MlirToRuntimeTranslator(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

  void Run() { EmitFunctions(); }

  virtual ~MlirToRuntimeTranslator();

 protected:
  //! Emit a "cinn.constant.*" operation, return true if succeed.
  bool EmitConstantOp(mlir::Operation* op);
  //! Emit a "cinn.return" operation.
  bool EmitReturnOp(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>* results);
  //! Emit a "ts.build_shape" operation.
  bool EmitBuildShapeOp(mlir::Operation* op);
  //! Emit an operation other than the special cases above.
  bool EmitGeneralOp(mlir::Operation* op);
  //! Emit all the functions.
  bool EmitFunctions();

  //! Emit a single function, this is an API that should be implemented by inherients.
  virtual void EmitFunction(mlir::FuncOp op);

  bool EmitCallOp(mlir::Operation* op, function_table_t* function_table);

  template <typename T>
  std::optional<T> EmitAttribute(const mlir::Attribute* attr);

  Value* GetOpResult(mlir::Operation* op);

  Value* GetValue(mlir::Value value);

  Value* AddValue(mlir::Value value);

  Value* AddValue(mlir::Value mlir_value, Value* value);

  void UpdateCurFuncName(std::string_view name);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Build a CoreRuntime from a MLIR module.
 */
void MlirToRuntimeTranslate(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

/**
 * Execute a MLIR program, that is execute all the functions without input arguments.
 * This is mainly used by testcase.
 * @param module a MLIR module.
 * @param registry the kernel registry containing all the valid kernels.
 */
void TestMlir(mlir::ModuleOp module, KernelRegistry* registry);

}  // namespace cinnrt::host_context
