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

Operation ComputeOp::Make(const std::string &name,
                          const std::string &tag,
                          const std::map<std::string, IrNodeRef> &attrs,
                          const std::vector<Var> &axis,
                          const std::vector<Expr> &body,
                          const std::vector<Expr> &shape) {
  auto n   = make_shared<ComputeOp>();
  n->name  = name;
  n->tag   = tag;
  n->attrs = attrs;
  n->axis  = axis;
  n->body  = body;
  n->shape = shape;
  return Operation(n);
}

Operation ComputeOp::Make(const std::string &name,
                          const std::string &tag,
                          const std::map<std::string, IrNodeRef> &attrs,
                          ComputeOp::handle_t handle,
                          const std::vector<Expr> &shape,
                          const std::vector<Expr> &domain,
                          const std::vector<Var> &reduce_axis) {
  auto n         = make_shared<ComputeOp>();
  n->name        = name;
  n->tag         = tag;
  n->attrs       = attrs;
  n->producer_fn = handle;
  n->shape       = domain;
  n->reduce_axis = reduce_axis;
  auto axis      = common::GenDefaultAxis(shape.size());
  std::vector<Expr> _axis;
  for (auto &x : axis) _axis.push_back(x);
  n->body = {handle(_axis)};
  return Operation(n);
}

}  // namespace ir
}  // namespace cinn
