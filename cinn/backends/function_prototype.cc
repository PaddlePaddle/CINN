#include "cinn/backends/function_prototype.h"

namespace cinn {
namespace backends {

bool FunctionProto::Match(const ir::Call *op) const {
  if (name != op->name) return false;
  if (ret_type != op->type()) return false;
  if (op->read_args.size() != readonly_arg_types.size()) return false;
  if (op->write_args.size() != mutable_arg_types.size()) return false;

  for (int i = 0; i < op->read_args.size(); i++) {
    if (op->read_args[i].type() != readonly_arg_types[i]) return false;
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    if (op->write_args[i].type() != mutable_arg_types[i]) return false;
  }
  return true;
}

void FunctionProto::AssertMatch(const ir::Call *op) const {
  CHECK_EQ(name, op->name);
  CHECK_EQ(ret_type, op->type());
  CHECK_EQ(op->read_args.size(), readonly_arg_types.size());
  CHECK_EQ(op->write_args.size(), mutable_arg_types.size());

  for (int i = 0; i < op->read_args.size(); i++) {
    CHECK_EQ(op->read_args[i].type(), readonly_arg_types[i]);
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    CHECK_EQ(op->write_args[i].type(), mutable_arg_types[i]);
  }
}

}  // namespace backends
}  // namespace cinn
