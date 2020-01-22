#include "cinn/ir/function.h"

namespace cinn {
namespace ir {

Args::Args(Value *values, int *type_codes, int len) {
  for (int i = 0; i < len; i++) {
    values_.emplace_back(values[i], type_codes[i]);
  }
}

}  // namespace ir
}  // namespace cinn
