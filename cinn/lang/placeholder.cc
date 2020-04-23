#include "cinn/lang/placeholder.h"

#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace lang {

ir::Tensor CreatePlaceHolder(const std::vector<Expr> &shape, Type type, const std::string &name) {
  if (type == Float(32)) {
    return Placeholder<float>(name, shape);
  } else if (type == Float(64)) {
    return Placeholder<double>(name, shape);
  } else if (type == Int(32)) {
    return Placeholder<int32_t>(name, shape);
  }
  NOT_IMPLEMENTED
}

}  // namespace lang
}  // namespace cinn
