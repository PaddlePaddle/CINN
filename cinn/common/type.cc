#include "cinn/common/type.h"

namespace cinn {
namespace common {

std::ostream &operator<<(std::ostream &os, const Type &t) {
  switch (t.type()) {
    case Type::type_t::Int:
      if (t.bits() == 1) {
        os << "bool";
      } else {
        os << "int" << t.bits();
      }

      break;
    case Type::type_t::UInt:
      os << "uint" << t.bits();
      break;

    case Type::type_t::Float:
      os << "float" << t.bits();
      break;
    default:
      LOG(FATAL) << "Unknown data type found";
  }

  if (t.width() > 1) os << "<" << t.width() << ">";
  if (t.is_cpp_handle()) os << "*";

  return os;
}

std::ostream &operator<<(std::ostream &os, Type::type_t t) {
  switch (t) {
    case Type::type_t::Void:
      os << "Void";
      break;
    case Type::type_t::UInt:
      os << "UInt";
      break;
    case Type::type_t::Int:
      os << "Int";
      break;
    case Type::type_t::Float:
      os << "Float";
      break;
    case Type::type_t::Unk:
      os << "Unk";
      break;
  }
  return os;
}

void Type::set_cpp_handle(bool x) {
  auto &v = (*reinterpret_cast<uint8_t *>(&cpp_type_));
  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::Handle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
}

Type Type::VectorOf(int w) const {
  CheckTypeValid();
  return Type(type_, w, bits_);
}

Type Type::ElementOf() const {
  CheckTypeValid();
  return Type(type_, bits_, 1);
}

void Type::CheckTypeValid() const { CHECK_NE(type_, type_t::Unk); }

Type Type::PointerOf() const {
  auto x = ElementOf();
  x.set_cpp_handle();
  return x;
}

}  // namespace common
}  // namespace cinn
