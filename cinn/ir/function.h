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
  Args() = default;
  Args(Value* values, int* type_codes, int len);

  void Append(Value v, int type_code) { values_.emplace_back(v, type_code); }

  size_t size() { return values_.size(); }
  //! Get i-th element.
  ArgValue operator[](int i) { return values_[i]; }

 private:
  std::vector<ArgValue> values_;
};

namespace detail {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void Run(const F& f, T&& value, Args&&... args) {
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, I + 1, F>::Run(f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void Run(const F& f) {}
};

template <typename F, typename... Args>
inline void for_each(const F& f, Args&&... args) {
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::Run(f, std::forward<Args>(args)...);
}

struct FuncArgsSetter {
  FuncArgsSetter(Args* args) : args_(args) {}

  template <typename T>
  void operator()(size_t I, T v) const {
    args_->Append(common::ToValue(v), PODValue::TypeCode<T>());
  }

 private:
  mutable Args* args_{};
};

}  // namespace detail

class PackedFunc {
 public:
  using func_t = std::function<void(Args args, RetValue*)>;

  PackedFunc() = default;
  explicit PackedFunc(const std::string& name) : name_(name) {}
  explicit PackedFunc(func_t body) : body_(body) {}

  template <typename... Args_>
  inline RetValue operator()(Args_&&... args) const {
    Args _args;
    detail::FuncArgsSetter setter(&_args);
    detail::for_each(setter, std::forward<Args_>(args)...);

    RetValue ret_value;
    body_(_args, &ret_value);
    return ret_value;
  }

 private:
  std::string name_;
  func_t body_;
};

}  // namespace ir
}  // namespace cinn
