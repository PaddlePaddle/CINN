#include "cinn/ir/lowered_func.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/ir.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/optim/tensor_write_tell.h"
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
  n->PrepareAllocOutputBufferExprs();
  n->AllocTempBuffer();
  n->PrepareBufferCastExprs();
  n->PrepareArgumentExprs();
  n->PrepareDeallocOutputBufferExprs();
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

void _LoweredFunc_::PrepareAllocOutputBufferExprs() {
  CHECK(alloc_output_buffer_exprs.empty()) << "duplicate prepare the allocate buffer for outputs";

  std::set<std::string> buffer_names;
  for (auto& arg : args) {
    if (arg.is_output()) {
      CHECK(arg.type().valid()) << "argument [" << arg.name() << "]'s type should be set";
      if (arg.is_buffer() && !buffer_names.count(arg.name())) {  // only buffer need allocation.
        buffer_names.insert(arg.name());                         // Avoid duplicate
        alloc_output_buffer_exprs.push_back(
            Alloc::Make(arg.buffer_arg(), arg.buffer_arg()->type(), arg.buffer_arg()->shape, Expr(), Expr()));
      }
    }
  }
}

void _LoweredFunc_::PrepareDeallocOutputBufferExprs() {
  CHECK(dealloc_output_buffer_exprs.empty()) << "duplicate prepare the allocate buffer for outputs";

  std::set<std::string> buffer_names;
  for (auto& arg : args) {
    if (arg.is_output()) {
      CHECK(arg.type().valid()) << "argument [" << arg.name() << "]'s type should be set";
      if (arg.is_buffer() && !buffer_names.count(arg.name())) {  // only buffer need allocation.
        buffer_names.insert(arg.name());                         // Avoid duplicate
        dealloc_output_buffer_exprs.push_back(Free::Make(arg.buffer_arg()));
      }
    }
  }
}

void _LoweredFunc_::AllocTempBuffer() {}

void _LoweredFunc_::PrepareBufferCastExprs() {
  // collect write.
  optim::TensorWriteTeller write_teller;
  write_teller.Collect(&body);

  auto tensors = CollectAllTensorReference();
  std::sort(tensors.begin(), tensors.end(), [](const Tensor& a, const Tensor& b) { return a->name < b->name; });
  VLOG(3) << "Function used " << tensors.size() << " buffers";
  for (auto& tensor : tensors) {
    auto* node = tensor.As<ir::_Tensor_>();
    CHECK(node);

    Type value_type = tensor->type().ElementOf();
    bool is_const   = !write_teller.IsWrite(tensor->name);
    value_type.set_as_cpp_handle();
    value_type.set_cpp_const(is_const);
    Var variable = _Var_::Make(tensor->name, value_type);

    Expr body = runtime::BufferGetDataHandle(tensor->buffer, is_const);

    auto let = Let::Make(variable, body);

    buffer_data_cast_exprs.push_back(let);
  }
}

void _LoweredFunc_::PrepareArgumentExprs() {
  // type of cinn_buffer_t**
  Type buffer_array_type;
  buffer_array_type.set_customized_type(common::customized_type::kbuffer_t);
  buffer_array_type.set_as_cpp_handle_handle();

  Var args_passed_in("_args", type_of<void*>());

  // get something like: cinn_buffer_t** args = (cinn_buffer_t**)(_args)
  Var array("args", buffer_array_type);
  {
    Expr body = Cast::Make(buffer_array_type, args_passed_in);
    argument_prepare_exprs.push_back(Let::Make(array, body));
  }

  /*
   * Get something like:
   *
   * const cinn_buffer_t* _A = args[0];
   * cinn_buffer_t* _B = args[1];
   */

  // Type of cinn_buffer_t*
  Type buffer_ptr_type;
  buffer_ptr_type.set_customized_type(common::customized_type::kbuffer_t);
  buffer_ptr_type.set_as_cpp_handle();

  // Type of const cinn_buffer_t*
  Type const_buffer_ptr_type = buffer_ptr_type.with_cpp_const();

  // We just has two kinds of argument types, first is `cinn_buffer_t*`, second is `const cinn_buffer_t*`, do not need a
  // `any` type support currently.
  for (int i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    Var _arg;
    if (arg.is_input())
      _arg = Var(arg.name(), const_buffer_ptr_type);
    else if (arg.is_output())
      _arg = Var(arg.name(), buffer_ptr_type);
    else
      NOT_IMPLEMENTED
    // currently we only support one type, cinn_buffer_t*
    Expr load_expr = Load::Make(array, {common::make_const(i)});
    Expr let_expr  = Let::Make(_arg, load_expr);
    argument_prepare_exprs.push_back(let_expr);
  }
}

std::vector<Tensor> _LoweredFunc_::CollectAllTensorReference() const {
  std::set<Expr> tensor_exprs = ir::CollectIRNodes(body, [](const Expr* expr) { return expr->As<ir::_Tensor_>(); });

  std::vector<Tensor> tensors;
  // remove the duplicate tensor by their name.
  std::set<std::string> names;

  for (const Expr& expr : tensor_exprs) {
    Expr& _expr = *const_cast<Expr*>(&expr);
    Tensor b(_expr.As<_Tensor_>());
    if (names.count(b->name)) continue;
    tensors.push_back(b);
    names.insert(b->name);
  }

  return tensors;
}

ir::Buffer Argument::buffer_arg() const {
  CHECK(is_buffer());
  return buffer_arg_;
}

ir::Var Argument::scalar_arg() const {
  CHECK(is_scalar());
  return scalar_arg_;
}

void Argument::set_buffer(const ir::Buffer& x) {
  kind        = Kind::kBuffer;
  buffer_arg_ = x;
}

void Argument::set_scalar(const ir::Var& x) {
  kind        = Kind::kScalar;
  scalar_arg_ = x;
}

}  // namespace ir
}  // namespace cinn
