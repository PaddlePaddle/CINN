#pragma once
#include <glog/logging.h>

#include <string>

#include <memory>
#include "cinn/common/macros.h"
#include "cinn/runtime/cinn_runtime.h"

//! Much of the concepts are borrowed from Halide project.

namespace cinn {
namespace common {

/**
 * Types in the CINN type system. They can be ints, unsigned ints, or floats of various bit-widths.
 * They can also be vectors of the same (by setting the `lanes` field to something larger than one).
 * NOTE: Front-end code other than vectorize shouldn't use vector types.
 */
struct Type {
  enum class type_t {
    Unk = -1,
    Int,
    UInt,
    Float,
    String,
    Void,
    // stupid idea to mix the Customized with other primitive types, large refactor needs here.
    Customized,  // Customized type
  };

  //! type decorators in C++, the different code can used together.
  enum class cpp_type_t : uint8_t {
    None         = 0,       // None information.
    Const        = 1,       // const.
    Handle       = 1 << 1,  // pointer type, such as `cinn_buffer_t*`.
    HandleHandle = 1 << 2,  // pointer of pointer, such as `cinn_buffer_t**`.
  };

  Type();
  Type(type_t t, int b, int w);
  Type(const Type& other);
  Type& operator=(const Type& other);

  bool is_primitive() const;
  bool is_customized() const;
  bool valid() const;

  //! Some helper functions to check a type.
  // @{
  CINN_NODISCARD bool is_unk() const;
  CINN_NODISCARD bool is_void() const;
  CINN_NODISCARD bool is_bool() const;
  CINN_NODISCARD bool is_vector() const;
  CINN_NODISCARD bool is_scalar() const;
  CINN_NODISCARD bool is_float(int bits = -1) const;
  CINN_NODISCARD bool is_int(int bits = -1) const;
  CINN_NODISCARD bool is_uint(int bits = -1) const;
  CINN_NODISCARD bool is_string() const;
  CINN_NODISCARD bool is_index_type();
  // @}

  Type& set_cpp_handle(bool x = true);
  CINN_NODISCARD bool is_cpp_handle() const;

  Type& set_cpp_handle_handle(bool x = true);
  CINN_NODISCARD bool is_cpp_handle_handle() const;

  Type& set_cpp_const(bool is_const = true);
  CINN_NODISCARD bool is_cpp_const() const;

  Type& set_customized_type(const std::string& t);
  const std::string& customized_type() const;
  CINN_NODISCARD bool is_customized_type() const;

  // Get a new type with bits set to \p x.
  Type with_bits(int x) const;
  // Get a new type with type set to \p x.
  Type with_type(type_t x) const;
  // Get a new type with lanes set to \p x.
  Type with_lanes(int x) const;
  // Get a new type with cpp_const set to \p x.
  Type with_cpp_const(bool x = true) const;

  //! Getters
  // @{
  type_t type() const;
  int bits() const;
  int lanes() const;
  cpp_type_t cpp_type() const;
  // @}

  //! Compare two types for equality.
  bool operator==(const Type& other) const;

  //! Compare two types for inequality.
  bool operator!=(const Type& other) const { return !(*this == other); }

  //! Generate a vector of this type, with `w` elements.
  Type VectorOf(int w) const;
  //! Generate a element type of this type.
  Type ElementOf() const;
  //! Generate the address type.
  Type PointerOf() const;

  friend std::ostream& operator<<(std::ostream& os, const Type& t);

  ~Type();

 private:
  void CheckTypeValid() const;

  struct Storage;
  Storage& GetStorage();
  const Storage& GetStorage() const;

  std::unique_ptr<Storage> storage_;
};  // namespace common

inline Type Void() { return Type(Type::type_t ::Void, 1, 0); }
inline Type Int(int bits, int lanes = 1) { return Type(Type::type_t ::Int, bits, lanes); }
inline Type UInt(int bits, int lanes = 1) { return Type(Type::type_t ::UInt, bits, lanes); }
inline Type Float(int bits, int lanes = 1) { return Type(Type::type_t ::Float, bits, lanes); }
inline Type Bool(int lanes = 1) { return Type(Type::type_t ::UInt, 1, lanes); }
inline Type String() { return Type(Type::type_t::String, 1, 1); }

template <typename T>
Type type_of();

// clang-format off
template <> inline Type type_of<float>() { return Float(32); }
template <> inline Type type_of<double>() { return Float(64); }
template <> inline Type type_of<unsigned char>() { return UInt(8); }
template <> inline Type type_of<int16_t>() { return UInt(16); }
template <> inline Type type_of<int32_t>() { return Int(32); }
template <> inline Type type_of<uint32_t>() { return UInt(32); }
template <> inline Type type_of<bool>() { return Bool(); }
template <> inline Type type_of<char>() { return Int(8); }
template <> inline Type type_of<int64_t>() { return Int(64); }
template <> inline Type type_of<uint64_t>() { return UInt(64); }
template <> inline Type type_of<signed char>() { return Int(8); }
template <> inline Type type_of<void>() { return Void(); }
// clang-format on
template <>
inline Type type_of<int8_t*>() {
  Type x = Int(8);
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<void*>() {
  Type x = type_of<void>();
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<void**>() {
  Type x = type_of<void>();
  x.set_cpp_handle_handle();
  return x;
}
template <>
inline Type type_of<float*>() {
  Type x = type_of<float>();
  x.set_cpp_handle();
  return x;
}
template <>
inline Type type_of<double*>() {
  Type x = type_of<double>();
  x.set_cpp_handle();
  return x;
}

std::ostream& operator<<(std::ostream& os, Type::type_t t);

namespace customized_type {

static const char* kArgs_type_repr     = "Args";
static const char* kArgValue_type_repr = "ArgValue";
static const char* kbuffer_t           = "cinn_buffer_t";
static const char* kpod_value_t        = "cinn_pod_value_t";

}  // namespace customized_type

template <>
inline Type type_of<cinn_buffer_t>() {
  return Type().set_customized_type(customized_type::kbuffer_t);
}
template <>
inline Type type_of<cinn_buffer_t*>() {
  return Type().set_customized_type(customized_type::kbuffer_t).set_cpp_handle();
}
template <>
inline Type type_of<const cinn_buffer_t*>() {
  return Type().set_customized_type(customized_type::kbuffer_t).set_cpp_handle().set_cpp_const();
}
template <>
inline Type type_of<cinn_pod_value_t>() {
  return Type().set_customized_type(customized_type::kpod_value_t);
}
template <>
inline Type type_of<cinn_pod_value_t*>() {
  return Type().set_customized_type(customized_type::kpod_value_t).set_cpp_handle();
}

}  // namespace common
}  // namespace cinn
