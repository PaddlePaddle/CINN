#include "cinn/common/context.h"

namespace cinn {
namespace common {

Context &Context::Global() {
  static Context x;
  return x;
}
}  // namespace common
}  // namespace cinn
