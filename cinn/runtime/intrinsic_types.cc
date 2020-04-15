#include "cinn/runtime/intrinsic_types.h"

namespace cinn {
namespace runtime {

Type BufferType::cinn_type() {
  Type type;
  type.set_customized_type(c_type_repr);
  return type;
}

char BufferType::c_type_repr[] = "cinn_buffer_t";

}  // namespace runtime
}  // namespace cinn
