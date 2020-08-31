#include "cinn/frontend/paddle/op_desc.h"

namespace cinn {
namespace frontend {
namespace paddle {

std::string OpDescReadAPI::Repr() const {
  std::stringstream ss;
  ss << Type();
  ss << "(";
  for (auto& arg : InputArgumentNames()) {
    ss << arg << ":";
    for (auto val : Input(arg)) {
      ss << val << " ";
    }
  }
  ss << ") -> (";
  for (auto& arg : OutputArgumentNames()) {
    ss << arg << ":";
    for (auto val : Output(arg)) {
      ss << val << " ";
    }
  }
  ss << ")";
  return ss.str();
}

}  // namespace paddle
}  // namespace frontend
}  // namespace cinn
