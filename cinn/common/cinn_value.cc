#include "cinn/common/cinn_value.h"

#include "cinn/ir/ir.h"
#include "cinn/ir/node.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {

namespace ir {

class Expr;
class Var;

}  // namespace ir

namespace common {

//! Implement the type_code for all the supported types.
// @{
#define __m(T, code__)           \
  template <>                    \
  int CINNValue::TypeCode<T>() { \
    return code__;               \
  }
__m(std::nullptr_t, -1);
__m(char *, 20);  // start from a larger number to avoid duplicate id with cinn_pod_value_t
__m(char const *, 21);
__m(ir::Expr, 22);
__m(ir::Var, 23);
__m(CINNValuePackShared, 24);
#undef __m
//@}

//! Implement ToValue.
// @{
template <>
cinn_value_t ToValue<int>(int v) {
  cinn_value_t val;
  val.v_int64 = v;
  return val;
}
template <>
cinn_value_t ToValue<int64_t>(int64_t v) {
  cinn_value_t val;
  val.v_int64 = v;
  return val;
}
template <>
cinn_value_t ToValue<float>(float v) {
  cinn_value_t val;
  val.v_float64 = v;
  return val;
}
template <>
cinn_value_t ToValue<double>(double v) {
  cinn_value_t val;
  val.v_float64 = v;
  return val;
}
template <>
cinn_value_t ToValue<char *>(char *v) {
  cinn_value_t val;
  val.v_str = v;
  return val;
}
template <>
cinn_value_t ToValue<char const *>(char const *v) {
  cinn_value_t val;
  val.v_str = const_cast<char *>(v);
  return val;
}
// @}

CINNValue::operator ir::Var() const {
  CHECK_EQ(type_code_, TypeCode<ir::Var>());
  return std::any_cast<ir::Var>(shared_);
}
CINNValue::operator ir::Expr() const {
  CHECK_EQ(type_code_, TypeCode<ir::Expr>());
  return std::any_cast<Expr>(shared_);
}
CINNValue::operator CINNValuePackShared() const {
  CHECK_EQ(type_code_, TypeCode<CINNValuePackShared>());
  return std::any_cast<CINNValuePackShared>(shared_);
}
CINNValue::CINNValue(char *value) : cinn_pod_value_t(ToValue(value), TypeCode<char *>()) {}

CINNValue::CINNValue(const Var &value) : cinn_pod_value_t(cinn_value_t(), TypeCode<Var>()) {
  CHECK(value.defined());
  shared_ = value;
}
CINNValue::CINNValue(const Expr &value) : cinn_pod_value_t(cinn_value_t(), TypeCode<Expr>()) {
  CHECK(value.defined());
  shared_ = value;
}
CINNValue::CINNValue(const CINNValuePackShared &value)
    : cinn_pod_value_t(cinn_value_t(), TypeCode<CINNValuePackShared>()) {
  CHECK(value.defined());
  shared_ = value;
}

CINNValuePackShared CINNValuePack::Make(const std::vector<CINNValue> &array) {
  auto *node = new CINNValuePack;
  for (auto &item : array) node->AddValue(item);
  return CINNValuePackShared(node);
}
CINNValue &CINNValuePack::operator[](int offset) {
  CHECK_LT(offset, size());
  return values_[offset];
}
const CINNValue &CINNValuePack::operator[](int offset) const {
  CHECK_LT(offset, size());
  return values_[offset];
}
void CINNValuePack::AddValue(const CINNValue &value) {
  CHECK(value.defined());
  values_.push_back(value);
}
void CINNValuePack::Clear() { values_.clear(); }
const char *CINNValuePack::type_info() const { return __type_info__; }

CINNValue &CINNValue::operator=(int32_t value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(int64_t value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(float value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(double value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(char *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(cinn_buffer_t *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(void *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const char *value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const CINNValuePackShared &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const ir::Var &value) {
  *this = CINNValue(value);
  return *this;
}
CINNValue &CINNValue::operator=(const ir::Expr &value) {
  *this = CINNValue(value);
  return *this;
}
}  // namespace common
}  // namespace cinn
