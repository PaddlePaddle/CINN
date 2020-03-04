#include "cinn/runtime/intrinsic.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace runtime {

ir::Expr BufferCreate(ir::Buffer buffer) {
  std::vector<Expr> args;
  args.push_back(buffer->data);
  CHECK(buffer->target.defined()) << "Buffer's target should be set before compile";
  return ir::Call::Make(Void(),
                        runtime::buffer_create,
                        {buffer->data, Expr(buffer->target.runtime_arch())},
                        ir::Call::CallType::Intrinsic);
  return ir::Expr();
}

ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr> &indices) {
  std::vector<ir::Expr> args({ir::Expr(buffer)});
  args.insert(std::end(args), indices.begin(), indices.end());

  return ir::Call::Make(           //
      buffer->type().ElementOf(),  //
      runtime::buffer_load,        //
      args,                        //
      ir::Call::Halide);
}

ir::Expr BufferMalloc(ir::Buffer buffer) { return BufferMalloc(buffer->data); }
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

}  // namespace runtime
}  // namespace cinn
