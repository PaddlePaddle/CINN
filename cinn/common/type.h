#pragma once
#include <glog/logging.h>

//! Much of the concepts are borrowed from Halide project.

namespace cinn {
namespace common {

/**
 * Types in the CINN type system. They can be ints, unsigned ints, or floats of various bit-widths.
 * They can also be vectors of the same (by setting the `width` field to something larger than one).
 * NOTE: Front-end code other than vectorize shouldn't use vector types.
 */
struct Type {
  enum class type_t {
    Unk = -1,
    Int,
    UInt,
    Float,
    Void,
  };

  //! type decorators in C++.
  enum class cpp_type_t : uint8_t {
    None   = 0,       // None information.
    Handle = 1,       // pointer type.
    Const  = 1 << 2,  // const
  };

  Type() = default;
  Type(type_t t, int b, int w) : type_(t), bits_(b), width_(w) {}

  //! Some helper functions to tell a type.
  // @{
  bool valid() const { return !is_unk(); }
  bool is_unk() const { return type_ == type_t::Unk; }
  bool is_void() const { return type_ == type_t::Void; }
  bool is_bool() const { return type_ == type_t::UInt && bits_ == 1; }
  bool is_vector() const { return width_ > 1; }
  bool is_scalar() const { return width_ == 1; }
  bool is_float(int bits = -1) const { return type_ == type_t::Float && (bits < 0 || bits == this->bits()); }
  bool is_int(int bits = -1) const { return type_ == type_t::Int && (bits < 0 || bits == this->bits()); }
  bool is_uint(int bits = -1) const { return type_ == type_t::UInt && (bits < 0 || bits == this->bits()); }
  // @}

  void set_as_cpp_handle(bool x = true);
  bool is_cpp_handle() const { return static_cast<uint8_t>(cpp_type_) & static_cast<uint8_t>(cpp_type_t::Handle); }

  //! Getters
  // @{
  type_t type() const { return type_; }
  int bits() const { return bits_; }
  int width() const { return width_; }
  cpp_type_t cpp_type() const { return cpp_type_; }
  // @}

  //! Compare two types for equality.
  bool operator==(const Type& other) const {
    return type_ == other.type_ && bits_ == other.bits_ && width_ == other.width_ && cpp_type_ == other.cpp_type_;
  }

  //! Compare two types for inequality.
  bool operator!=(const Type& other) const { return !(*this == other); }

  //! Generate a vector of this type, with `w` elements.
  Type VectorOf(int w) const;
  //! Generate a element type of this type.
  Type ElementOf() const;
  //! Generate the address type.
  Type PointerOf() const;

  friend std::ostream& operator<<(std::ostream& os, const Type& t);

 private:
  void CheckTypeValid() const;

  type_t type_{type_t::Unk};
  cpp_type_t cpp_type_{cpp_type_t::None};

  //! How many bits per element.
  int bits_{};

  //! How many elements(if a vector type), for scalar types, it should be 1.
  int width_{1};
};

inline Type Void() { return Type(Type::type_t ::Void, 0, 0); }
inline Type Int(int bits, int width = 1) { return Type(Type::type_t ::Int, bits, width); }
inline Type UInt(int bits, int width = 1) { return Type(Type::type_t ::UInt, bits, width); }
inline Type Float(int bits, int width = 1) { return Type(Type::type_t ::Float, bits, width); }
inline Type Bool(int width = 1) { return Type(Type::type_t ::UInt, 1, width); }

template <typename T>
Type type_of();

// clang-format off
template <> inline Type type_of<float>() { return Float(32); }
template <> inline Type type_of<double>() { return Float(64); }
template <> inline Type type_of<unsigned char>() { return UInt(8); }
template <> inline Type type_of<int16_t>() { return UInt(16); }
template <> inline Type type_of<unsigned int>() { return UInt(32); }
template <> inline Type type_of<bool>() { return Bool(); }
template <> inline Type type_of<char>() { return Int(8); }
template <> inline Type type_of<int>() { return Int(32); }
template <> inline Type type_of<signed char>() { return Int(8); }
template <> inline Type type_of<void>() { return Void(); }
// clang-format on
template <>
inline Type type_of<void*>() {
  Type x = type_of<void>();
  x.set_as_cpp_handle();
  return x;
}
template <>
inline Type type_of<float*>() {
  Type x = type_of<float>();
  x.set_as_cpp_handle();
  return x;
}
template <>
inline Type type_of<double*>() {
  Type x = type_of<double>();
  x.set_as_cpp_handle();
  return x;
}

std::ostream& operator<<(std::ostream& os, Type::type_t t);

}  // namespace common
}  // namespace cinn
