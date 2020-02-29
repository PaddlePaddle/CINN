#include "cinn/ir/lowered_func.h"

#include "cinn/common/common.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace ir {

const _LoweredFunc_* LoweredFunc::operator->() const { return As<_LoweredFunc_>(); }
_LoweredFunc_* LoweredFunc::operator->() { return As<_LoweredFunc_>(); }

LoweredFunc _LoweredFunc_::Make(const std::string& name, const std::vector<Argument>& args, const Expr& body) {
  auto* n = make_shared<_LoweredFunc_>();
  n->name = name;
  n->args = args;
  n->body = body;
  n->CheckValid();
  n->AllocBufferForOutputs();
  n->AllocTempBuffer();
  return LoweredFunc(n);
}

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const std::vector<Expr>& body) {
  CHECK_EQ(body.size(), 1);
  return Make(name, args, body.front());
}

void _LoweredFunc_ ::CheckValid() const {
  // check there is at least one output
  int out_count = 0;
  int in_count  = 0;
  for (auto& arg : args) {
    in_count += arg.is_input();
    out_count += arg.is_output();
  }
  CHECK_GT(out_count, 0) << "At least one output argument is needed for a function";
}

std::vector<Expr*> _LoweredFunc_::expr_fields() { return {&body}; }
std::vector<const Expr*> _LoweredFunc_::expr_fields() const { return {&body}; }

void _LoweredFunc_::AllocBufferForOutputs() {
  CHECK(alloc_output_buffer_exprs.empty()) << "duplicate prepare the allocate buffer for outputs";

  for (auto& arg : args) {
    if (arg.is_output()) {
      auto data = _Var_::Make(arg.name, arg.type);
      auto expr = Call::Make(Void(), runtime::buffer_alloc, {Expr(data)}, Call::CallType::Intrinsic);
      alloc_output_buffer_exprs.push_back(expr);
    }
  }
}

void _LoweredFunc_::AllocTempBuffer() {}

}  // namespace ir
}  // namespace cinn
