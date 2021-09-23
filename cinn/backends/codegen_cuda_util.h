#pragma once

#include <string>
#include <tuple>
#include "absl/container/flat_hash_map.h"
#include <vector>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace backends {

/**
 * Split a CINN Module into two separate modules, one cantains the host functions, the other contains the device
 * kernels.
 *
 * This contains some process:
 *
 * - replace the original kernel function with a Call node and add it to the first module, add a device kernel function
 * to the second module.
 */
std::tuple<ir::Module, ir::Module> SplitCudaAndHostModule(ir::Module module);

namespace detail {

struct CollectHostFunctionVisitor : public ir::IRMutator<> {
  explicit CollectHostFunctionVisitor(const std::string& module_name)
      : host_module_builder(module_name + "_host", common::DefaultHostTarget()),
        device_module_builder(module_name + "_gpu_device", common::DefaultNVGPUTarget()) {}

  std::tuple<ir::Module, ir::Module> operator()(Expr* expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return std::make_tuple(host_module_builder.Build(), device_module_builder.Build());
  }

 private:
  void Visit(const ir::_LoweredFunc_* op, Expr* expr) override {
    if (IsCudaFunction(op)) {
      CHECK(op->cuda_axis_info.valid());

      auto host_func = CreateHostFunctionGivenDeviceKernel(op);
      host_module_builder.AddFunction(host_func.as_lowered_func_ref());
      device_module_builder.AddFunction(CreateDeviceFunctionGivenDeviceKernel(*expr).as_lowered_func_ref());
    } else {
      host_module_builder.AddFunction(expr->as_lowered_func_ref());
    }
  }

  /**
   * Create a wrapper function for a kernel.
   *
   * For example, we get a kernel function:
   *
   * \code
   * __global__
   * void fn (float* a, float* out) { ... }
   * \endcode
   *
   * A host wrapper function will generate for it
   *
   * \code
   * void fn (cinn_buffer_t* a, cinn_buffer_t* out) {
   *   Call(fn_kernel);
   * }
   * \endcode
   */
  Expr CreateHostFunctionGivenDeviceKernel(const ir::_LoweredFunc_* func) {
    std::vector<Expr> args;
    // NOTE the suffix `__ptr` makes this argument lower to a pointer in LLVM backend.
    args.push_back(Var("args__ptr", type_of<cinn_pod_value_t*>()));
    args.push_back(Var("num_args", type_of<int32_t>()));

    auto call =
        ir::Call::Make(Void(), GenDeviceKernelName(func->name), args, {}, ir::CallType::Extern, ir::FunctionRef(), 0);
    Expr body = ir::Block::Make({call});

    std::vector<ir::Argument> host_func_args;
    host_func_args.emplace_back(args[0], ir::Argument::IO::kOutput);
    host_func_args.emplace_back(args[1], ir::Argument::IO::kOutput);
    auto host_func            = ir::_LoweredFunc_::Make(func->name, host_func_args, body, {});
    host_func->cuda_axis_info = func->cuda_axis_info;
    return host_func;
  }

  Expr CreateDeviceFunctionGivenDeviceKernel(Expr expr) {
    auto copied        = optim::IRCopy(expr);
    auto* lowered_func = copied.as_lowered_func();
    lowered_func->name = GenDeviceKernelName(lowered_func->name);
    return copied;
  }

  inline std::string GenDeviceKernelName(const std::string& fn) { return fn + "_kernel"; }

  bool IsCudaFunction(const ir::_LoweredFunc_* func);

 private:
  ir::Module::Builder host_module_builder;
  ir::Module::Builder device_module_builder;
};

}  // namespace detail

}  // namespace backends
}  // namespace cinn
