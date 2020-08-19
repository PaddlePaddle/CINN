#include "cinn/lang/packed_func.h"

namespace cinn {
namespace lang {

Args::Args(cinn_value_t *values, int *type_codes, int len) {
  for (int i = 0; i < len; i++) {
    values_.emplace_back(values[i], type_codes[i]);
  }
}

}  // namespace lang
}  // namespace cinn
