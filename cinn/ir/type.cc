#include "cinn/ir/type.h"

namespace cinn {
namespace ir {

std::ostream &operator<<(std::ostream &os, const Type &t) {
  switch (t.type()) {
    case Type::Int:
      if (t.bits() == 1) {
        os << "bool";
      } else {
        os << "int" << t.bits();
      }

      break;
    case Type::UInt:
      os << "uint" << t.bits();
      break;

    case Type::Float:
      os << "float" << t.bits();
      break;
    default:
      LOG(FATAL) << "Unknown data type found";
  }
  if (t.width() > 1) os << "<" << t.width() << ">";

  return os;
}

}  // namespace ir
}  // namespace cinn
