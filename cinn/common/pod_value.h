#pragma once
#include <glog/logging.h>
#include <boost/variant/get.hpp>
#include <boost/variant/variant.hpp>

#include "cinn/common/type.h"

namespace cinn {
namespace common {

union Value {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
};

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
  // @}

  //! Setters.
  // @{
  bool set_float(Value value, int type_code);
  bool set_double(Value value, int type_code);
  bool set_int32(Value value, int type_code);
  bool set_int64(Value value, int type_code);
  bool set_handler(Value value, int type_code);
  // @}

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
