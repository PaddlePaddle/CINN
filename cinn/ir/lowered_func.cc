#include "cinn/ir/lowered_func.h"

#include "cinn/common/common.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
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
  n->PrepareBufferCastExprs();
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
      CHECK(arg.type.valid()) << "argument's type is not defined";
      auto data = _Var_::Make(arg.name, arg.type);
      auto expr = runtime::BufferMalloc(data);
      alloc_output_buffer_exprs.push_back(expr);
    }
  }
}

void _LoweredFunc_::AllocTempBuffer() {}

void _LoweredFunc_::PrepareBufferCastExprs() {
  auto buffers = CollectAllBufferReference();
  VLOG(3) << "Function used " << buffers.size() << " buffers";
  for (auto& b : buffers) {
    auto* node = b->As<ir::_Buffer_>();
    CHECK(node);
    std::string buffer_name = b->name;
    std::string tensor_name = BufferGetTensorName(node);

    Type value_type = b->type().ElementOf();
    value_type.set_cpp_handle();
    Var value = _Var_::Make(tensor_name, value_type);

    Expr body = runtime::BufferGetDataHandle(b);

    auto let = Let::Make(value, body);

    buffer_data_cast_exprs.push_back(let);
  }
}

std::vector<Buffer> _LoweredFunc_::CollectAllBufferReference() {
  auto buffer_exprs = ir::CollectIRNodes(body, [](const Expr* expr) { return expr->As<ir::_Buffer_>(); });

  std::vector<Buffer> buffers;
  // remove the duplicate buffer by their name.
  std::set<std::string> names;

  for (auto& expr : buffer_exprs) {
    Buffer b(expr->As<_Buffer_>());
    if (names.count(b->name)) continue;
    buffers.push_back(b);
    names.insert(b->name);
  }

  return buffers;
}

}  // namespace ir
}  // namespace cinn
