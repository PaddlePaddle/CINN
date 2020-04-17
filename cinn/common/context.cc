#include "cinn/common/context.h"

#include "cinn/ir/ir.h"

namespace cinn {
namespace common {
using utils::any;

Context& Context::Global() {
  static Context x;
  return x;
}

}  // namespace common
}  // namespace cinn
