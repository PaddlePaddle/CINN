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

ir::Expr BufferMalloc(ir::Buffer buffer) {
  return ir::Call::Make(Void(), runtime::buffer_malloc, {buffer->data}, ir::Call::Intrinsic);
}

}  // namespace runtime
}  // namespace cinn
