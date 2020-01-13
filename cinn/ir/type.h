#pragma once
#include <glog/logging.h>

//! Much of the concepts are borrowed from Halide project.

namespace cinn {
namespace ir {

/**
 * Types in the CINN type system. They can be ints, unsigned ints, or floats of various bit-widths.
 * They can also be vectors of the same (by setting the `width` field to something larger than one).
 * NOTE: Front-end code other than vectorize shouldn't use vector types.
 */
struct Type {
  enum type_t {
    Unk = -1,
    Int,
    UInt,
    Float,
  };

  Type() = default;
  Type(type_t t, int b, int w) : type_(t), bits_(b), width_(w) {}

  //! Some helper functions to tell a type.
  // @{
  bool valid() const { return !is_unk(); }
  bool is_unk() const { return type_ == Unk; }
  bool is_bool() const { return type_ == UInt && bits_ == 1; }
  bool is_vector() const { return width_ > 1; }
  bool is_scalar() const { return width_ == 1; }
  bool is_float() const { return type_ == Float; }
  bool is_int() const { return type_ == Int; }
  bool is_uint() const { return type_ == UInt; }
  // @}

  //! Getters
  // @{
  type_t type() const { return type_; }
  int bits() const { return bits_; }
  int width() const { return width_; }
  // @}

  //! Compare two types for equality.
  bool operator==(const Type& other) const {
    return type_ == other.type_ && bits_ == other.bits_ && width_ == other.width_;
  }

  //! Compare two types for inequality.
  bool operator!=(const Type& other) const { return !(*this == other); }

  //! Generate a vector of this type, with `w` elements.
  Type VectorOf(int w) const {
    CheckTypeValid();
    return Type(type_, w, bits_);
  }

  //! Generate a element type of this type.
  Type ElementOf() const {
    CheckTypeValid();
    return Type(type_, bits_, 1);
  }

 private:
  void CheckTypeValid() const { CHECK_NE(type_, Unk); }

  type_t type_;

  //! How many bits per element.
  int bits_{};

  //! How many elements(if a vector type), for scalar types, it should be 1.
  int width_{1};
};

inline Type Int(int bits, int width = 1) { return Type(Type::Int, bits, width); }
inline Type UInt(int bits, int width = 1) { return Type(Type::UInt, bits, width); }
inline Type Float(int bits, int width = 1) { return Type(Type::Float, bits, width); }
inline Type Bool(int width = 1) { return Type(Type::UInt, 1, width); }

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
// clang-format on

}  // namespace ir
}  // namespace cinn
