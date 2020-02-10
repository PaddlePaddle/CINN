#include "cinn/runtime/intrinsic.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace runtime {

ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr> &indices) {
  std::vector<ir::Expr> args({ir::Expr(buffer)});
  args.insert(std::end(args), indices.begin(), indices.end());

  return ir::Call::Make(           //
      buffer->type().ElementOf(),  //
      runtime::buffer_load,        //
      args,                        //
      ir::Call::Extern);
}

}  // namespace runtime
}  // namespace cinn
