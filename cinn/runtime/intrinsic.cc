#include "cinn/runtime/intrinsic.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace runtime {

ir::Expr BufferCreate(ir::Buffer buffer) {
  std::vector<Expr> args;
  args.push_back(ir::_Var_::Make(buffer->name, buffer->type()));
  args.push_back(Expr(buffer->target.runtime_arch()));
  CHECK(buffer->target.defined()) << "Buffer's target should be set before compile";
  return ir::Call::Make(Void(), runtime::buffer_create, args, ir::Call::CallType::Intrinsic);
}

ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr> &indices) {
  std::vector<ir::Expr> args({ir::Expr(buffer->buffer_addr())});
  args.insert(std::end(args), indices.begin(), indices.end());

  if (!buffer->type().is_float()) {
    NOT_IMPLEMENTED
  }

  std::string buffer_load_method;
  if (buffer->type().bits() == 32)
    buffer_load_method = buffer_load_float32;
  else if (buffer->type().bits() == 64)
    buffer_load_method = buffer_load_float64;
  else {
    LOG(ERROR) << "support for type " << buffer->type() << " not implemented";
    NOT_IMPLEMENTED
  }

  return ir::Call::Make(           //
      buffer->type().ElementOf(),  //
      buffer_load_method,          //
      args,                        //
      ir::Call::Intrinsic);
}

ir::Expr BufferMalloc(ir::Buffer buffer) { return BufferMalloc(buffer->buffer_addr()); }
ir::Expr BufferMalloc(ir::Var buffer_var) {
  return ir::Call::Make(Void(), runtime::buffer_malloc, {Expr(0), buffer_var}, ir::Call::Intrinsic);
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
  }
  LOG(FATAL) << "Not supported type " << type;
  return cinn_unk_t();
}

ir::Expr BufferGetDataHandle(ir::Buffer buffer, bool is_mutable) {
  CHECK(buffer->type().valid());
  Type type = Void();
  type.set_as_cpp_handle();  // a void*
  auto call = ir::Call::Make(type, buffer_get_data_handle, {Expr(buffer)}, ir::Call::CallType::Intrinsic);

  Type target_type = buffer->type().ElementOf();
  target_type.set_as_cpp_handle();
  auto cast = ir::Cast::Make(target_type, call);
  return cast;
}

}  // namespace runtime
}  // namespace cinn
