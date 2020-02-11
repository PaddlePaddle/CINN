#include "cinn/ir/operation.h"

#include "cinn/common/common.h"

namespace cinn {
namespace ir {

Operation PlaceholderOp::Make(const std::string &name, const std::vector<Expr> &shape, Type dtype) {
  auto n   = make_shared<PlaceholderOp>();
  n->name  = name;
  n->shape = shape;
  n->set_type(dtype);
  return Operation(n);
}
const char *PlaceholderOp::func_type() const { return __func_type__; }

const char *ComputeOp::func_type() const { return __func_type__; }

Operation ComputeOp::Make(std::string name,
                          std::string tag,
                          std::map<std::string, IrNodeRef> attrs,
                          std::vector<Var> axis,
                          std::vector<Expr> body) {
  auto n   = make_shared<ComputeOp>();
  n->name  = name;
  n->tag   = tag;
  n->attrs = attrs;
  n->axis  = axis;
  n->body  = body;
  return Operation(n);
}

}  // namespace ir
}  // namespace cinn