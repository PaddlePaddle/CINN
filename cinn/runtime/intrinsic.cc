#include "cinn/runtime/intrinsic.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace runtime {

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

Expr IntrinsicCall(Type type,
                   const std::string& fn_name,
                   const std::vector<Expr>& args,
                   const std::vector<Expr>& write_args) {
  return ir::Call::Make(type, fn_name, args, write_args, ir::CallType::Intrinsic, ir::FunctionRef(), 0);
}

}  // namespace runtime
}  // namespace cinn
