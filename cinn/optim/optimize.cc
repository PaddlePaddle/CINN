#include "cinn/optim/optimize.h"

#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/remove_nested_block.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/optim/transform_polyfor_to_for.h"
#include "cinn/optim/unroll_loops.h"
#include "cinn/optim/vectorize_loops.h"

namespace cinn {
namespace optim {

Expr Optimize(Expr e) {
  CHECK(e.defined());
  auto copied = IRCopy(e);

  TransformPolyForToFor(&copied);
  Simplify(&copied);
  VectorizeLoops(&copied, Target());
  UnrollLoop(&copied);
  RemoveGpuForloopsAxis(&copied);
  RemoveNestedBlock(&copied);
  Simplify(&copied);

  return copied;
}

}  // namespace optim
}  // namespace cinn
