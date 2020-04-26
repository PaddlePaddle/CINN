#include "cinn/ir/operation.h"
#include <memory>
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

const char *PlaceholderOp::func_type() const { return "placeholder_op"; }

const char *ComputeOp::func_type() const { return "compute_op"; }

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

Operation CallOp::Make(const std::string &call_target,
                       const std::vector<Expr> &arg_list,
                       int value_slot,
                       Expr call_op) {
  auto n         = make_shared<CallOp>();
  n->arg_list    = arg_list;
  n->call_target = call_target;
  n->arg_slot    = value_slot;
  n->call_expr   = call_op;
  return Operation(n);
}

const char *ComputeOp::__func_type__     = "compute_op";
const char *PlaceholderOp::__func_type__ = "placeholder_op";
const char *CallOp::__func_type__        = "call_op";

const char *CallOp::func_type() const { return __func_type__; }

}  // namespace ir
}  // namespace cinn
