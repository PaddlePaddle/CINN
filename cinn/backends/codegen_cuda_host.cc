#include "cinn/backends/codegen_cuda_host.h"

#include <algorithm>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

const int kArgsArrayMaxLen = 20;

llvm::Value* CodeGenCUDA_Host::LowerGPUKernelLauncher(const ir::_LoweredFunc_* func) {
  CHECK(func->cuda_axis_info.valid());
  LOG(INFO)<<"Func name is : " << func->name;
  LOG(INFO) <<"Its cuda axis info is : " << func->cuda_axis_info;
  /* The current function definiton is
   * void fn(cinn_pod_value_t* args, int num_args) {
   *     Call(fn_kernel, args, num_args);
   * }
   * will lower to
   * void fn(cinn_pod_value_t* args, int num_args) { // num_args is ignored here.
   *    // NOTE the num_args is unnecessary here, but it should follow the pattern of CINN function.
   *    cinn_call_cuda_kernel(fn_kernel_ptr, args, grid_dim, block_dim, fn_kernel_stream);
   * }
   *
   * NOTE the global variables related to CUDA in LLVM module are
   * 1. fn_kernel_ptr, the pointer to the compiled kernel function returned by CUDA driver
   * 2. fn_kernel_stream, the CUDA stream this kernel should launch on.
   */

  // hard-code here to verify it is a simple CUDA host function.
  // @{
  auto body   = func->body;
  auto* block = body.As<ir::Block>();
  CHECK(block);

  CHECK_EQ(block->stmts.size(), 1UL);
  auto* call = block->stmts[0].As<ir::Call>();
  CHECK(call);
  // @}

  // Create the function
  // @{
  auto* function_type      = GenFunctionTypeFromCinnFunction(func, true);
  llvm::Function* function = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage, func->name, m_);
  function->setCallingConv(llvm::CallingConv::C);
  function->setHasUWTable();

  std::vector<llvm::Value*> ll_function_args;
  std::transform(function->arg_begin(), function->arg_end(), std::back_inserter(ll_function_args), [](auto& arg) {
    return std::addressof(arg);
  });

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function,
      /*InsertBefore=*/nullptr);
  b_->SetInsertPoint(entry);
  // @}

  // Get the arguments of the function.
  // @{
  auto* ll_args       = ll_function_args[0];
  auto* ll_args_count = ll_function_args[1];
  CHECK_EQ(ll_args->getType(), ll_cinn_pod_p_ty());   // cinn_pod_value_t* args
  CHECK_EQ(ll_args_count->getType(), ll_int32_ty());  // int32

  auto* ll_num_args_copied = b_->CreateAlloca(ll_int32_ty(), nullptr, "num_args_copied");
  Store(ll_args_count, ll_num_args_copied);
  SetVar(std::string(ll_num_args_copied->getName()), ll_num_args_copied);

  const std::string& func_arg0_name = func->args[0].name();
  CHECK(LLVM_WillVarLowerAsPointer(func_arg0_name))
      << "Variable [" << func_arg0_name << "] should have a name like someting will be lower to a pointer";
  SetVar(func->args[0].var_arg()->name, ll_args);
  // @}

  const std::string kernel_ptr_global_var_name    = GenKernelPtrVarName(func->name);
  const std::string kernel_stream_global_var_name = GenKernelStreamVarName(func->name);
  // set global variables to reference the [kernel_ptr] and [kernel_stream] for this kernel
  SetVar(kernel_ptr_global_var_name, m_->getOrInsertGlobal(kernel_ptr_global_var_name, ll_void_p_ty()));
  SetVar(kernel_stream_global_var_name, m_->getOrInsertGlobal(kernel_stream_global_var_name, ll_void_p_ty()));

  {  // create a new Call node for the ExternFunctionEmitter
    Var args_var(func->args[0].var_arg()->name, type_of<cinn_pod_value_t*>());  // pass *args directly to kernel
    Var kernel_fn_ptr_var(kernel_ptr_global_var_name, type_of<void*>());
    Var kernel_stream_var(kernel_stream_global_var_name, type_of<void*>());

    auto new_call_node = ir::Call::Make(Void(),
                                        runtime::intrinsic::call_cuda_kernel,
                                        {
                                            kernel_fn_ptr_var,  // kernel_fn
                                            args_var,           // args
                                            Var(std::string(ll_num_args_copied->getName()), type_of<int32_t>()),
                                            Expr(func->cuda_axis_info.grid_dim(0)),   // grid_x
                                            Expr(func->cuda_axis_info.grid_dim(1)),   // grid_y
                                            Expr(func->cuda_axis_info.grid_dim(2)),   // grid_z
                                            Expr(func->cuda_axis_info.block_dim(0)),  // block_x
                                            Expr(func->cuda_axis_info.block_dim(1)),  // block_y
                                            Expr(func->cuda_axis_info.block_dim(2)),  // block_z
                                            kernel_stream_var                         // stream
                                        },
                                        {},
                                        ir::CallType::Extern,
                                        ir::FunctionRef(),
                                        0);

    auto emitter_id = ExternFuncID{backend_llvm_host, runtime::intrinsic::call_cuda_kernel};
    auto* emitter   = ExternFunctionEmitterRegistry::Global().Lookup(emitter_id);
    CHECK(emitter) << "No extern function emitter called " << emitter_id;
    emitter->BindCodeGen(this);
    emitter->Emit(new_call_node.As<ir::Call>());
  }

  RetVoid();

  return function;
}

}  // namespace backends
}  // namespace cinn
