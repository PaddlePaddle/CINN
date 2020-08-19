#include "cinn/ir/ir_visitor.h"

#include <unordered_set>

#include "cinn/ir/ir_printer.h"
#include "cinn/lang/tensor.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

bool operator==(Expr a, Expr b) {
  if (a.get() == b.get()) return true;
  // TODO(Superjomn) implement with a more accurate one
  return utils::GetStreamCnt(a) == utils::GetStreamCnt(b);
}

bool operator!=(Expr a, Expr b) { return !(a == b); }

}  // namespace ir
}  // namespace cinn
