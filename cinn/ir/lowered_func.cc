#include "cinn/ir/lowered_func.h"

#include <algorithm>
#include <iostream>
#include <memory>
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

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const Expr& body,
                                const std::vector<ir::Buffer>& temp_bufs) {
  auto* n      = make_shared<_LoweredFunc_>();
  n->name      = name;
  n->args      = args;
  n->body      = body;
  n->temp_bufs = temp_bufs;

  n->CheckValid();
  n->PrepareAllocOutputBufferExprs();
  n->PrepareAllocTempBufferExprs();
  n->AllocTempBuffer();
  n->PrepareBufferCastExprs();
  n->PrepareArgumentExprs();
  n->PrepareDeallocOutputBufferExprs();
  return LoweredFunc(n);
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

void _LoweredFunc_::PrepareAllocTempBufferExprs() {
  for (auto& temp_buf : temp_bufs) {
    alloc_tmp_buffer_exprs.push_back(Alloc::Make(temp_buf, temp_buf->type(), temp_buf->shape, Expr(), Expr()));
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
    value_type.set_cpp_handle();
    value_type.set_cpp_const(is_const);
    Var variable = _Var_::Make(tensor->name, value_type);

    Expr body = runtime::BufferGetDataHandle(tensor->buffer, is_const);

    auto let = Let::Make(variable, body);

    buffer_data_cast_exprs.push_back(let);
  }
}

void _LoweredFunc_::PrepareArgumentExprs() {
  // type of `void*`
  auto void_ptr_array_type = Type().with_type(Type::type_t::Void).set_cpp_handle();
  // type of `cinn_buffer_t*`
  auto buffer_ptr_type = Type().set_customized_type(common::customized_type::kbuffer_t).set_cpp_handle();
  // type of `const cinn_buffer_t*`
  auto const_buffer_ptr_type = buffer_ptr_type.with_cpp_const();
  CHECK(!buffer_ptr_type.is_cpp_const());

  Var args_passed_in("_args", type_of<void*>());
  auto pod_value_ptr = common::CastIfNeeded(args_passed_in, type_of<cinn_pod_value_t*>());

  if (FLAGS_cinn_runtime_display_debug_info) {
    argument_prepare_exprs.push_back(runtime::IntrinsicCall(
        Void(), runtime::print_debug_args_repr, {pod_value_ptr, common::make_const(Int(32), args.size())}));
  }

  /*
   * Get something like:
   *
   * const cinn_buffer_t* _A = args[0];
   * cinn_buffer_t* _B = (cinn_buffer_t*)args[1];
   * int M = (int)arg[2];
   */

  // We just has two kinds of argument types, first is `cinn_buffer_t*`, second is `const cinn_buffer_t*`, do not need a
  // `any` type support currently.
  for (int i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    // cast arg to cinn_pod_value_t*

    // something like `_args[0]`
    Expr load_expr = Load::Make(pod_value_ptr, {common::make_const(i)});

    Var _arg;
    bool is_const = arg.is_input();

    if (arg.is_buffer()) {
      auto buffer_type = is_const ? const_buffer_ptr_type : buffer_ptr_type;
      _arg             = Var(arg.name(), buffer_type);
    } else if (arg.is_var()) {
      _arg = Var(arg.name(), arg.var_arg()->type());
    } else {
      NOT_IMPLEMENTED
    }

    CHECK(_arg->type().valid());

    Expr pod_cast_expr;

    if (arg.is_buffer()) {
      pod_cast_expr = runtime::IntrinsicCall(arg.type(), runtime::pod_value_to_buffer_p, {load_expr});
    } else if (arg.type() == type_of<int32_t>()) {
      pod_cast_expr = runtime::IntrinsicCall(arg.type(), runtime::pod_value_to_int32, {load_expr});
    } else if (arg.type() == type_of<int64_t>()) {
      pod_cast_expr = runtime::IntrinsicCall(arg.type(), runtime::pod_value_to_int64, {load_expr});
    } else if (arg.type() == type_of<float>()) {
      pod_cast_expr = runtime::IntrinsicCall(arg.type(), runtime::pod_value_to_float, {load_expr});
    } else if (arg.type() == type_of<double>()) {
      pod_cast_expr = runtime::IntrinsicCall(arg.type(), runtime::pod_value_to_double, {load_expr});
    } else {
      NOT_IMPLEMENTED
    }

    Expr let_expr = Let::Make(_arg, pod_cast_expr);
    CHECK(let_expr.type().valid());
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

ir::Var Argument::var_arg() const {
  CHECK(is_var());
  return var_arg_;
}

void Argument::set_buffer(const ir::Buffer& x) {
  CHECK(!is_var()) << "the buffer is already a var";
  buffer_arg_ = x;
}

void Argument::set_var(const ir::Var& x) {
  CHECK(!is_buffer()) << "the buffer is already a buffer";
  var_arg_ = x;
}

Argument::Argument(const ir::Buffer& buffer, Argument::IO io) {
  set_buffer(buffer);
  this->io = io;
}

Type Argument::type() const {
  if (is_var())
    return var_arg()->type();
  else if (is_buffer())
    return buffer_arg()->type();
  else
    NOT_IMPLEMENTED
}

std::string Argument::name() const {
  if (is_buffer())
    return buffer_arg()->name;
  else if (is_var())
    return var_arg()->name;
  else
    NOT_IMPLEMENTED
  return "";
}

Argument::Argument(const ir::Var& var, Argument::IO io) {
  set_var(var);
  this->io = io;
}

std::string Argument::human_readable() const {
  std::stringstream os;
  os << "<Argument: " << name() << " ";
  os << (is_input() ? "R" : "W");
  os << ">";
  return os.str();
}

}  // namespace ir
}  // namespace cinn
