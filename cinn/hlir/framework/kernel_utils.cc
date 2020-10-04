#include "cinn/hlir/framework/kernel_utils.h"

namespace cinn::hlir::framework {

const char* AnyValue::type_info() const { return __type_info__; }

}  // namespace cinn::hlir::framework
