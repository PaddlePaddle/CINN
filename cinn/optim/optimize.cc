#include "cinn/optim/optimize.h"

#include "cinn/ir/ir_printer.h"
#include "cinn/optim/activate_to_extern_call.h"
#include "cinn/optim/cache_read_write_replace.h"
#include "cinn/optim/call_arg_list_to_pod_value.h"
#include "cinn/optim/eliminate_broadcast_in_forloop.h"
#include "cinn/optim/extern_call_process.h"
#include "cinn/optim/fold_cinn_call_arguments.h"
#include "cinn/optim/insert_debug_log_callee.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/lower_function_call_bind_vars.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/optim/transform_polyfor_to_for.h"
#include "cinn/optim/unroll_loops.h"
#include "cinn/optim/vectorize_loops.h"

namespace cinn {
namespace optim {

Expr Optimize(Expr e, bool runtime_debug_info) {
  CHECK(e.defined());
  auto copied = IRCopy(e);

  FoldCINNCallArguments(&copied);
  TransformPolyForToFor(&copied);
  Simplify(&copied);
  VectorizeLoops(&copied, Target());
  EliminateBroadcastInForloop(&copied);
  UnrollLoop(&copied);
#ifdef CINN_WITH_CUDA
  RemoveGpuForloopsAxis(&copied);
  CudaSyncThreadsDropIfThenElse(&copied);
#endif
  // CacheReadWriteReplace(&copied);

  RemoveNestedBlock(&copied);

  ActivateToExternCall(&copied);
  ExternCallMultiOutputShallowStore(&copied);

  Simplify(&copied);

  if (runtime_debug_info) {
    LOG(WARNING) << "Turn on runtime debug information output";
    InsertDebugLogCallee(&copied);
  }

  return copied;
}

lang::Module Optimize(const lang::Module& module) {
  auto copied = IRCopy(Expr(module));

  LowerFunctionCallBindVars(&copied);
  CallArgListToPodValue(&copied);

  return copied.as_module_ref();
}

}  // namespace optim
}  // namespace cinn
