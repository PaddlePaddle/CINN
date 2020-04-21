#pragma once
#include <glog/logging.h>
#include <vector>
#include "cinn/common/common.h"
#include "cinn/common/macros.h"
#include "cinn/common/object.h"
#include "cinn/common/type.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/utils/any.h"

struct cinn_buffer_t;

namespace cinn {

namespace ir {

class Expr;
class Var;

}  // namespace ir

namespace common {

template <typename T>
cinn_value_t ToValue(T v);

class CINNValue;
class CINNValuePackShared;

/**
 * A CINNValuePack is a shared Array of multiple CINNValue.
 */
struct CINNValuePack : public common::Object {
  /**
   * Create a new CINNValuePack instance.
   * @param array The list of CINNValues.
   * @return a CINNValuePack.
   */
  static CINNValuePackShared Make(const std::vector<CINNValue>& array);

  //! Get i-th element in mutable mode.
  CINNValue& operator[](int offset);
  //! Get i-th element in readonly mode.
  const CINNValue& operator[](int offset) const;

  //! Add one \p value to the tail.
  void AddValue(const CINNValue& value);

  //! Remove all the values.
  void Clear();

  size_t size() const { return values_.size(); }

  CINN_DISALLOW_COPY_AND_ASSIGN(CINNValuePack);

  const char* type_info() const override;

 private:
  CINNValuePack() = default;
  std::vector<CINNValue> values_;
};

struct CINNValuePackShared : public Shared<CINNValuePack> {
  CINNValuePackShared(CINNValuePack* ptr) : Shared<CINNValuePack>(ptr) {}

  CINNValue& operator[](int offset) { return (*operator->())[offset]; }
  const CINNValue& operator[](int offset) const { return (*operator->())[offset]; }

  CINNValuePack* operator->() { return get(); }
  const CINNValuePack* operator->() const { return get(); }
};

/**
 * Handler for value types in CINN system. It supports two kinds of values: the POD and Shared.
 */
class CINNValue : public cinn_pod_value_t {
 public:
  static constexpr int kNull = -1;

  CINNValue() : cinn_pod_value_t(cinn_value_t(), kNull) {}
  CINNValue(cinn_value_t value, int type_code) : cinn_pod_value_t(value, type_code) {}

  explicit CINNValue(int32_t value) : cinn_pod_value_t(value) {}
  explicit CINNValue(int64_t value) : cinn_pod_value_t(value) {}
  explicit CINNValue(float value) : cinn_pod_value_t(value) {}
  explicit CINNValue(double value) : cinn_pod_value_t(value) {}
  explicit CINNValue(char* value);
  explicit CINNValue(cinn_buffer_t* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(void* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(const char* value) : cinn_pod_value_t(value) {}
  explicit CINNValue(ir::Var value);
  explicit CINNValue(ir::Expr value);
  explicit CINNValue(const CINNValuePackShared& value);

  bool defined() const { return type_code_ != kNull; }

  //! The value getters for the supported types.
  // @{
  using cinn_pod_value_t::operator double;
  using cinn_pod_value_t::operator float;
  using cinn_pod_value_t::operator int32_t;
  using cinn_pod_value_t::operator int64_t;
  using cinn_pod_value_t::operator void*;
  using cinn_pod_value_t::operator cinn_buffer_t*;
  using cinn_pod_value_t::operator char*;
  operator ir::Var() const;
  operator ir::Expr() const;
  operator CINNValuePackShared() const;
  // @}

  //! Assign operators
  // @{
  CINNValue& operator=(int32_t value);
  CINNValue& operator=(int64_t value);
  CINNValue& operator=(float value);
  CINNValue& operator=(double value);
  CINNValue& operator=(char* value);
  CINNValue& operator=(const ir::Var& value);
  CINNValue& operator=(const ir::Expr& value);
  CINNValue& operator=(cinn_buffer_t* value);
  CINNValue& operator=(void* value);
  CINNValue& operator=(const CINNValuePackShared& value);
  CINNValue& operator=(const char* value);
  // @}

  //! Set the value.
  template <typename T>
  void Set(T v);

  /**
   * Get the type code for a specific POD type.
   * @param T some data type.
   * @return an integer representing the type code.
   */
  template <typename T>
  static int TypeCode();

 protected:
  utils::any shared_;
};

}  // namespace common
}  // namespace cinn
