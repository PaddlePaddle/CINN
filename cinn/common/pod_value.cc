#include "cinn/common/pod_value.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"

namespace cinn {
namespace common {

//! Implement the type_code for all the supported types.
// @{
#define __m(T, code__)          \
  template <>                   \
  int PODValue::TypeCode<T>() { \
    return code__;              \
  }
__m(std::nullptr_t, -1);
__m(int, 0);
__m(int64_t, 1);
__m(float, 2);
__m(double, 3);
__m(void *, 4);
__m(char *, 5);
__m(char const *, 5);
__m(ir::Expr, 4);
__m(ir::Var, 4);
#undef __m
//@}

//! Implement setters.
// @{
// @}

PODValue::operator double() const {
  CHECK_EQ(TypeCode<double>(), type_code_);
  return value_.v_float64;
}
PODValue::operator float() const {
  CHECK_EQ(TypeCode<float>(), type_code_);
  return value_.v_float64;
}
PODValue::operator int32_t() const {
  CHECK_EQ(TypeCode<int32_t>(), type_code_);
  return value_.v_int64;
}
PODValue::operator int64_t() const {
  CHECK_EQ(TypeCode<int64_t>(), type_code_);
  return value_.v_int64;
}
PODValue::operator void *() const {
  CHECK_EQ(TypeCode<void *>(), type_code_);
  return value_.v_handle;
}
PODValue::operator char *() const {
  CHECK_EQ(TypeCode<char *>(), type_code_);
  return value_.v_str;
}

// Value setter for multiple types.
// @{
template <>
void PODValue::Set<int32_t>(int32_t v) {
  type_code_     = TypeCode<int32_t>();
  value_.v_int64 = v;
}
template <>
void PODValue::Set<int64_t>(int64_t v) {
  type_code_     = TypeCode<int64_t>();
  value_.v_int64 = v;
}
template <>
void PODValue::Set<float>(float v) {
  type_code_       = TypeCode<float>();
  value_.v_float64 = v;
}
template <>
void PODValue::Set<double>(double v) {
  type_code_       = TypeCode<double>();
  value_.v_float64 = v;
}
template <>
void PODValue::Set<void *>(void *v) {
  type_code_      = TypeCode<void *>();
  value_.v_handle = v;
}
template <>
void PODValue::Set<char *>(char *v) {
  type_code_   = TypeCode<char *>();
  value_.v_str = v;
}
template <>
void PODValue::Set<char const *>(char const *v) {
  type_code_   = TypeCode<char *>();
  value_.v_str = const_cast<char *>(v);
}
template <>
void PODValue::Set<ir::Var>(ir::Var v) {
  type_code_      = TypeCode<ir::Var>();
  value_.v_handle = v.ptr();
}
template <>
void PODValue::Set<ir::Expr>(ir::Expr v) {
  type_code_      = TypeCode<ir::Expr>();
  value_.v_handle = v.ptr();
}
// @}

//! Implement ToValue.
// @{
template <>
Value ToValue<int>(int v) {
  Value val;
  val.v_int64 = v;
  return val;
}
template <>
Value ToValue<int64_t>(int64_t v) {
  Value val;
  val.v_int64 = v;
  return val;
}
template <>
Value ToValue<float>(float v) {
  Value val;
  val.v_float64 = v;
  return val;
}
template <>
Value ToValue<double>(double v) {
  Value val;
  val.v_float64 = v;
  return val;
}
template <>
Value ToValue<char *>(char *v) {
  Value val;
  val.v_str = v;
  return val;
}
template <>
Value ToValue<char const *>(char const *v) {
  Value val;
  val.v_str = const_cast<char *>(v);
  return val;
}
template <>
Value ToValue<ir::Expr>(ir::Expr v) {
  Value val;
  val.v_handle = v.ptr();
  return val;
}
template <>
Value ToValue<ir::Var>(ir::Var v) {
  Value val;
  val.v_handle = v.ptr();
  return val;
}
// @}

}  // namespace common
}  // namespace cinn
