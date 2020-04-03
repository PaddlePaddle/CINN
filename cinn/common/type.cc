#include "cinn/common/type.h"

namespace cinn {
namespace common {

std::ostream &operator<<(std::ostream &os, const Type &t) {
  if (t.is_cpp_const()) os << "const ";
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
    case Type::type_t::Void:
      os << "void";
      break;
    case Type::type_t::Customized:
      os << t.customized_type();
      break;
    default:
      LOG(FATAL) << "Unknown data type found";
  }

  if (t.lanes() > 1) os << "<" << t.lanes() << ">";
  if (t.is_cpp_handle()) os << "*";
  if (t.is_cpp_handle_handle()) os << "**";

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
    case Type::type_t::Customized:
      os << "Customized";
  }
  return os;
}

void Type::set_as_cpp_handle(bool x) {
  auto &v = (*reinterpret_cast<uint8_t *>(&cpp_type_));
  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::Handle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::Handle);
}

void Type::set_as_cpp_handle_handle(bool x) {
  auto &v = (*reinterpret_cast<uint8_t *>(&cpp_type_));
  if (x)
    v |= static_cast<uint8_t>(cpp_type_t::HandleHandle);
  else
    v &= ~static_cast<uint8_t>(cpp_type_t::HandleHandle);
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
  x.set_as_cpp_handle();
  return x;
}

Type Type::with_bits(int x) const {
  Type type  = *this;
  type.bits_ = x;
  return type;
}

Type Type::with_type(Type::type_t x) const {
  Type type  = *this;
  type.type_ = x;
  return type;
}

Type Type::with_lanes(int x) const {
  Type type   = *this;
  type.lanes_ = x;
  return type;
}

Type Type::with_cpp_const(bool x) const {
  Type type = *this;
  type.set_cpp_const(x);
  return type;
}

void Type::set_cpp_const(bool is_const) {
  uint8_t &data = *reinterpret_cast<uint8_t *>(&cpp_type_);
  if (is_const) {
    data |= static_cast<uint8_t>(cpp_type_t::Const);
  } else {
    data &= ~(static_cast<uint8_t>(cpp_type_t::Const));
  }
}

}  // namespace common
}  // namespace cinn
