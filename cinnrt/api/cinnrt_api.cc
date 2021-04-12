#include "cinnrt/api/cinnrt_api.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/mlir_loader.h"
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"
#include "cinnrt/host_context/value.h"
#include "cinnrt/kernel/basic_kernels.h"
#include "cinnrt/kernel/control_flow_kernels.h"
#include "cinnrt/kernel/tensor_kernels.h"
#include "cinnrt/kernel/tensor_shape_kernels.h"
#include "cinnrt/kernel/test_kernels.h"
#include "cinnrt/tensor/tensor_map.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace cinnrt::host_context;
using namespace cinnrt::tensor;

namespace cinnrt {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

struct MlirToRuntimeTranslator::Impl {
  mlir::ModuleOp module;
  // The runtime for a function call.
  CoreRuntimeBuilder* runtime{};
  // The current working op, the translator process the ops one by one, each time it updates `cur_op` here to current op
  // working on.
  OpExecutableBuilder* cur_op{};

  // record the current function name.
  std::string cur_func_name;

  // Name to function definitions.
  std::unordered_map<std::string, mlir::FuncOp> func_defs;

  // Map from an operation to its results.
  std::unordered_map<const mlir::Operation*, std::vector<ValueRef>> op_results;
  llvm::DenseMap<mlir::Value, ValueRef> value_map;
};

/**
 * Execute the mlir program in predict mode -- print some debug infomation to stdout.
 */
class MlirPredictExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder core_runtime;

  MlirPredictExecutor(mlir::ModuleOp module, KernelRegistry* registry)
      : core_runtime(registry), MlirToRuntimeTranslator(module, &core_runtime), registry(registry) {
    CHECK(registry);
  }

  void Run() {
    EmitFunctions();

    CHECK(registry);
    for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
      VLOG(3) << "Running function " << func_op.getName().str();
      if (func_op.getName().str() != "predict") continue;
      EmitAndRunFuncWithoutArguments(func_op);
    }
  }

 protected:
  std::unordered_map<std::string, mlir::FuncOp> func_def_table;

  void EmitFunction(mlir::FuncOp op) override {
    auto it = impl_->func_defs.try_emplace(op.getName().str(), op);
    CHECK(it.second) << "Duplicate function defition found for function [" << op.getName().str();
  }

 private:
  void EmitAndRunFuncWithoutArguments(mlir::FuncOp func) {
    // print the function name for llvm FileChecker macro, CHECK-LABEL
    std::cout << '@' << func.getName().str() << std::endl;
    if (func.getNumArguments() == 0) {  // an entry function, execute it immediately
      VLOG(3) << "executing function " << func.getName().str();
      // Emit and execute each function
      CoreRuntimeBuilder runtime(registry);
      impl_->runtime = &runtime;

      auto& blocks = func.getBlocks();
      CHECK_EQ(blocks.size(), 1UL) << "function with more than one block is not supported yet";

      for (auto& op : blocks.front()) {
        if (EmitConstantOp(&op)) continue;
        if (EmitBuildShapeOp(&op)) continue;
        llvm::SmallVector<mlir::Value, 3> results;
        if (EmitReturnOp(&op, &results)) continue;
        if (EmitCallOp(&op, &impl_->func_defs)) continue;
        if (EmitGeneralOp(&op)) continue;
        LOG(FATAL) << "Not supported op: " << DumpToString(op);
      }

      runtime.Execute();

    } else {
      VLOG(2) << "get an callable function: " << func.getName().str();
    }
  }

 private:
  KernelRegistry* registry{};
};

std::shared_ptr<CinnrtPredictor> CreateCinnrtPredictor(const CinnrtConfig& config) {
  auto x = std::make_shared<CinnrtPredictor>();
  x->Init(config);
  return x;
}

struct CinnrtPredictor::Impl {
  mlir::OwningModuleRef module_ref;
  KernelRegistry* registry;
  TensorMap* map;
};

CinnrtPredictor::CinnrtPredictor() : impl_(new Impl) {}
CinnrtPredictor::~CinnrtPredictor() {}

void CinnrtPredictor::Run() {
  // std::cout << "CinnrtPredictor::Run" << std::endl;
  MlirPredictExecutor execute(impl_->module_ref.get(), impl_->registry);
  execute.Run();
}

int CinnrtPredictor::Init(const CinnrtConfig& config) {
  mlir::MLIRContext* context = cinnrt::Global::getMLIRContext();
  auto module                = dialect::LoadMlirFile(config.mlir_path(), context);

  KernelRegistry* registry = new KernelRegistry();

  kernel::RegisterBasicKernels(registry);
  kernel::RegisterTestKernels(registry);
  kernel::RegisterTensorShapeKernels(registry);
  kernel::RegisterTensorKernels(registry);
  kernel::RegisterControlFlowKernels(registry);

  impl_->module_ref = std::move(module);
  impl_->registry   = registry;

  // load extra shared library
  for (const std::string& lib_path : config.shared_libs()) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      return 1;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func = reinterpret_cast<void (*)(host_context::KernelRegistry*)>(reg_sym);
      reg_func(registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path << "\". Skip.\n";
    }
  }
  // Load params
  TensorMap* map = new TensorMap();
  impl_->map     = map;

  return 0;
}

}  // namespace cinnrt
