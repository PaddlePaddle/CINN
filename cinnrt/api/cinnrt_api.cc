#include "cinnrt/api/cinnrt_api.h"

#include <vector>

#include "cinnrt/common/global.h"
#include "cinnrt/dialect/dense_tensor.h"
#include "cinnrt/dialect/mlir_loader.h"
#include "cinnrt/host_context/core_runtime.h"
#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/mlir_function_executable.h"
#include "cinnrt/host_context/mlir_to_runtime_translate.h"
#include "cinnrt/host_context/op_executable.h"
#include "cinnrt/host_context/value.h"
#include "cinnrt/kernel/basic_kernels.h"
#include "cinnrt/kernel/control_flow_kernels.h"
#include "cinnrt/kernel/tensor_kernels.h"
#include "cinnrt/kernel/tensor_shape_kernels.h"
#include "cinnrt/kernel/test_kernels.h"
#include "cinnrt/tensor/tensor_map.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"

using namespace cinnrt::host_context;
using namespace cinnrt::tensor;
using namespace cinnrt::tensor;
using cinnrt::dt::TensorMapType;
using cinnrt::dt::TensorType;

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
 * Execute the mlir program in predict mode.
 */
class PredictExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder core_runtime;

  PredictExecutor(mlir::ModuleOp module, KernelRegistry* registry, TensorMap* map)
      : core_runtime(registry), MlirToRuntimeTranslator(module, &core_runtime), registry_(registry) {
    CHECK(registry_);
    Init(map);
  }

  void Run() {
    auto arguments = llvm::makeArrayRef(arguments_);
    auto results   = llvm::makeMutableArrayRef(results_.begin(), results_.size());
    function_executable_->Execute(arguments, results);
  }

  int GetInputNum() { return inputs_.size(); }

  DenseHostTensor* GetInput(int i) { return inputs_[i]; }

  int GetOutputNum() { return outputs_.size(); }

  DenseHostTensor* GetOutput(int i) { return outputs_[i]; }

 private:
  void Init(TensorMap* map) {
    EmitFunctions();
    llvm::Optional<mlir::FuncOp> predict_func_ = llvm::None;
    for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
      if (func_op.getName().str() != "predict") continue;
      predict_func_ = func_op;
      break;
    }
    if (!predict_func_) {
      std::cout << "ERROR: init failed, no predict function found in mlir." << std::endl;
      return;
    }
    auto& predict_func   = predict_func_.getValue();
    function_executable_ = new MlirFunctionExecutable(predict_func, registry_, impl_->func_defs);

    // process parammeters
    for (int i = 0; i < predict_func.getNumArguments(); ++i) {
      auto arg  = predict_func.getArgument(i);
      auto type = arg.getType();
      // this param is TensorMap
      if (type.isa<TensorMapType>()) {
        auto* value = new host_context::Value(std::move(*map));
        arguments_.push_back(value);
        AddValue(predict_func.getArgument(i), value);
      } else {
        // this param is an input Tensor
        auto dht    = DenseHostTensor();
        auto* value = new host_context::Value(std::move(dht));
        arguments_.push_back(value);
        inputs_.push_back(&(value->get<DenseHostTensor>()));
      }
    }

    // process results
    auto& last_op = predict_func.front().back();
    if (last_op.getName().getStringRef() == "cinn.return") {
      for (int i = 0; i < last_op.getNumOperands(); ++i) {
        auto* value = AddValue(mlir::Value(last_op.getOperand(i)));
        results_.push_back(ValueRef(value));
        outputs_.push_back(&(value->get<DenseHostTensor>()));
      }
    }
  }

 protected:
  std::unordered_map<std::string, mlir::FuncOp> func_def_table;

  void EmitFunction(mlir::FuncOp op) override {
    auto it = impl_->func_defs.try_emplace(op.getName().str(), op);
    CHECK(it.second) << "Duplicate function defition found for function [" << op.getName().str();
  }

 private:
  KernelRegistry* registry_{};
  MlirFunctionExecutable* function_executable_;
  llvm::SmallVector<DenseHostTensor*, 1> inputs_;
  llvm::SmallVector<host_context::Value*, 2> arguments_;
  llvm::SmallVector<DenseHostTensor*, 1> outputs_;
  llvm::SmallVector<ValueRef, 1> results_;
};

std::shared_ptr<CinnRtPredictor> CreateCinnRtPredictor(const CinnRtConfig& config) {
  auto x = std::make_shared<CinnRtPredictor>();
  x->Init(config);
  return x;
}

struct CinnRtPredictor::Impl {
  mlir::OwningModuleRef module_ref;
  PredictExecutor* executor;
};

CinnRtPredictor::CinnRtPredictor() : impl_(new Impl) {}
CinnRtPredictor::~CinnRtPredictor() {}

void CinnRtPredictor::Run() { impl_->executor->Run(); }

int CinnRtPredictor::Init(const CinnRtConfig& config) {
  mlir::MLIRContext* context = cinnrt::Global::getMLIRContext();
  auto module_ref            = dialect::LoadMlirFile(config.mlir_path(), context);

  KernelRegistry* registry = new KernelRegistry();

  kernel::RegisterBasicKernels(registry);
  kernel::RegisterTestKernels(registry);
  kernel::RegisterTensorShapeKernels(registry);
  kernel::RegisterTensorKernels(registry);
  kernel::RegisterControlFlowKernels(registry);

  impl_->module_ref = std::move(module_ref);

  // load extra shared library
  for (const std::string& lib_path : config.shared_libs()) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib = llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      return 1;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func = reinterpret_cast<void (*)(KernelRegistry*)>(reg_sym);
      reg_func(registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path << "\". Skip.\n";
    }
  }
  // Load params
  TensorMap* map = LoadParams(config.model_dir());
  // Create PredictExecutor
  impl_->executor = new PredictExecutor(impl_->module_ref.get(), registry, map);
  return 0;
}

int CinnRtPredictor::GetInputNum() { return impl_->executor->GetInputNum(); }

DenseHostTensor* CinnRtPredictor::GetInput(int i) { return impl_->executor->GetInput(i); }

int CinnRtPredictor::GetOutputNum() { return impl_->executor->GetOutputNum(); }

DenseHostTensor* CinnRtPredictor::GetOutput(int i) { return impl_->executor->GetOutput(i); }

}  // namespace cinnrt
