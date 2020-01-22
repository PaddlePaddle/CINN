#include "cinn/common/pod_value.h"

namespace cinn {
namespace common {

//! Implement the type_code for all the supported types.
// @{
#define __m(T, code__)          \
  template <>                   \
  int PODValue::TypeCode<T>() { \
    return code__;              \
  }
__m(int, 0);
__m(int64_t, 1);
__m(float, 2);
__m(double, 3);
__m(void *, 4);
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
};
PODValue::operator void *() const {
  CHECK_EQ(TypeCode<void *>(), type_code_);
  return value_.v_handle;
};

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
// @}

}  // namespace common
}  // namespace cinn
