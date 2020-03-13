#include "cinn/optim/optimize.h"

#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

Expr Optimize(Expr e) {
  auto copied = IRCopy(e);
  Simplify(&copied);

  return copied;
}

}  // namespace optim
}  // namespace cinn
