#include "cinn/ir/node.h"

namespace cinn {
namespace ir {

template <>
void ExprNode<IntImm>::Accept(cinn::ir::IRVisitor *v) const {}

}  // namespace ir
}  // namespace cinn
