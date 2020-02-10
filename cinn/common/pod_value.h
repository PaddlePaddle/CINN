#pragma once
#include <glog/logging.h>

#include <boost/variant/get.hpp>
#include <boost/variant/variant.hpp>

#include "cinn/common/type.h"

namespace cinn {

namespace ir {

class Expr;
class Var;

}  // namespace ir

namespace common {

union Value {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  char* v_str;
};

template <typename T>
Value ToValue(T v);

/**
 * Handler for several POD data types.
 */
class PODValue {
 public:
  static constexpr int kNull = -1;

  PODValue() : type_code_{kNull} {}
  PODValue(Value value, int type_code) : value_(value), type_code_(type_code) {}

  //! The value getters for the supported types.
  // @{
  operator double() const;
  operator float() const;
  operator int32_t() const;
  operator int64_t() const;
  operator void*() const;
  operator char*() const;
  operator ir::Var() const;
  operator ir::Expr() const;
  // @}

  //! Set the value.
  template <typename T>
  void Set(T v);

  int type_code() const { return type_code_; }

  /**
   * Get the type code for a specific POD type.
   * @tparam T some data type.
   * @return an integer representing the type code.
   */
  template <typename T>
  static int TypeCode();

 protected:
  int type_code_;
  Value value_;
};

}  // namespace common
}  // namespace cinn
