#include "cinn/runtime/intrinsic.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace runtime {

Expr GetAddr(Type type, Expr arg) { return IntrinsicCall(type, intrisic::get_address_repr, {arg}); }

ir::Expr BufferCreate(ir::Buffer buffer) {
  std::vector<Expr> args;
  args.push_back(Expr(buffer));
  args.push_back(Expr(buffer->target.runtime_arch()));
  CHECK(buffer->target.defined()) << "Buffer [" << buffer->name << "] target not set, get " << buffer->target;
  return ir::Call::Make(Void(), intrisic::buffer_create, args, {}, ir::CallType::Intrinsic, ir::FunctionRef(), 0);
}

ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr>& indices) {
  std::vector<ir::Expr> args({ir::Expr(buffer->buffer_addr())});
  args.insert(std::end(args), indices.begin(), indices.end());

  if (!buffer->type().is_float()) {
    CINN_NOT_IMPLEMENTED
  }

  std::string buffer_load_method;
  if (buffer->type().bits() == 32) {
    buffer_load_method = intrisic::buffer_load_float32;
  } else if (buffer->type().bits() == 64) {
    buffer_load_method = intrisic::buffer_load_float64;
  } else {
    LOG(ERROR) << "support for type " << buffer->type() << " not implemented";
    CINN_NOT_IMPLEMENTED
  }

  return ir::Call::Make(           //
      buffer->type().ElementOf(),  //
      buffer_load_method,          //
      args,
      {},
      ir::CallType::Intrinsic,
      ir::FunctionRef(),
      0);
}

ir::Expr BufferMalloc(ir::Buffer buffer) { return BufferMalloc(buffer->buffer_addr()); }
ir::Expr BufferMalloc(ir::Var buffer_var) {
  return ir::Call::Make(
      Void(), intrisic::buffer_malloc, {Expr(0), buffer_var}, {}, ir::CallType::Intrinsic, ir::FunctionRef(), 0);
}

cinn_type_t ToRuntimeType(Type type) {
  if (type == Int(32)) {
    return cinn_int32_t();
  } else if (type == Int(64)) {
    return cinn_int64_t();
  } else if (type == UInt(32)) {
    return cinn_uint64_t();
  } else if (type == Float(32)) {
    return cinn_float32_t();
  } else if (type == Float(64)) {
    return cinn_float64_t();
  } else if (type == Float(32).PointerOf()) {
    return cinn_type_of<float*>();
  }
  LOG(FATAL) << "Not supported type " << type;
  return cinn_unk_t();
}

ir::Expr BufferGetDataHandle(ir::Buffer buffer, bool is_const) {
  CHECK(buffer->type().valid());
  Type type = buffer->type();
  type.set_cpp_handle();
  type.set_cpp_const(is_const);
  Expr call;
  if (!is_const)
    call = ir::Call::Make(
        type, intrisic::buffer_get_data_handle, {Expr(buffer)}, {}, ir::CallType::Intrinsic, ir::FunctionRef(), 0);
  else
    call = ir::Call::Make(type,
                          intrisic::buffer_get_data_const_handle,
                          {Expr(buffer)},
                          {},
                          ir::CallType::Intrinsic,
                          ir::FunctionRef(),
                          0);

  Type target_type = buffer->type().ElementOf();
  target_type.set_cpp_handle();
  target_type.set_cpp_const(is_const);
  auto cast = ir::Cast::Make(target_type, call);
  return cast;
}

Expr IntrinsicCall(Type type,
                   const std::string& fn_name,
                   const std::vector<Expr>& args,
                   const std::vector<Expr>& write_args) {
  return ir::Call::Make(type, fn_name, args, write_args, ir::CallType::Intrinsic, ir::FunctionRef(), 0);
}

}  // namespace runtime
}  // namespace cinn
