#pragma once

#include <string>
#include <vector>

#include "cinn/common/pod_value.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {
using common::PODValue;
using common::Value;

/**
 * A single argument value to Function.
 */
class ArgValue : public common::PODValue {
 public:
  ArgValue() = default;

  ArgValue(common::Value value, int type_code) : common::PODValue(value, type_code) {}

  // Reuse coverter from parent.
  using common::PODValue::operator double;
  using common::PODValue::operator float;
  using common::PODValue::operator int32_t;
  using common::PODValue::operator int64_t;
};

/**
 * \brief Return value container.
 */
class RetValue : public PODValue {
 public:
  RetValue() = default;

  RetValue(RetValue&& other) : common::PODValue(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_      = kNull;
  }

  // Reuse converter from parent
  using common::PODValue::operator double;
  using common::PODValue::operator float;
  using common::PODValue::operator int32_t;
  using common::PODValue::operator int64_t;
};

class Args {
 public:
  Args(Value* values, int* type_codes, int len);
  size_t size() { return values_.size(); }
  //! Get i-th element.
  ArgValue operator[](int i) { return values_[i]; }

 private:
  std::vector<ArgValue> values_;
};

class PackedFunc {
 public:
  using func_t = std::function<void(Args args, RetValue*)>;

  PackedFunc() = default;
  explicit PackedFunc(const std::string& name) : name_(name) {}
  explicit PackedFunc(func_t body) : body_(body) {}

  template <typename... Args_>
  inline RetValue operator()(Args_&&... args) const {
    const int kNumArgs   = sizeof...(Args_);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    Value values[kArraySize];
    int type_codes[kArraySize];

    RetValue ret_value;
    body_(Args(values, type_codes, kNumArgs), &ret_value);
  }

 private:
  std::string name_;
  func_t body_;
};

}  // namespace ir
}  // namespace cinn
