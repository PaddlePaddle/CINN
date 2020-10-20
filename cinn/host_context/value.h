#pragma once
#include <glog/logging.h>
#include <utility>
#include <variant>
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/host_context/tensor_shape.h"

namespace cinn {
namespace host_context {

using ValueVariantType = std::variant<int32_t, int64_t, float, double, bool, TensorShape>;

/**
 * Represents any data type for value in host context.
 */
class Value : public common::Object {
 public:
  using variant_type = ValueVariantType;

  explicit Value() {}
  explicit Value(int32_t x) : data(x) {}
  explicit Value(int64_t x) : data(x) {}
  explicit Value(float x) : data(x) {}
  explicit Value(double x) : data(x) {}
  explicit Value(bool x) : data(x) {}
  explicit Value(TensorShape&& x) : data(std::move(x)) {}

  const char* type_info() const override;

  ValueVariantType data;

 private:
  static constexpr const char* __type_info__ = "host_context_value";
};

/**
 * Represents a counted reference of a Value.
 */
class ValueRef : common::Shared<Value> {
 public:
  ValueRef() = default;
  explicit ValueRef(Value* n) : common::Shared<Value>(n) {}
  explicit ValueRef(int32_t val);
  explicit ValueRef(int64_t val);
  explicit ValueRef(float val);
  explicit ValueRef(double val);
  explicit ValueRef(bool val);

  using common::Shared<Value>::get;
  using common::Shared<Value>::Reset;
  using common::Shared<Value>::operator->;
  using common::Shared<Value>::operator*;
  //! Get a readonly data.
  template <typename T>
  T get() const {
    CHECK(p_);
    return std::get<T>(p_->data);
  }

  //! Assign a data.
  template <typename T>
  void Assign(const T& x) {
    if (!p_) {
      p_ = common::make_shared<Value>();
    }
    *p_ = x;
  }

  template <typename T, typename... Args>
  void Assign(Args... args) {
    p_ = common::make_shared<T>(std::forward<Args>(args)...);
  }

  inline bool IsValid() { return p_; }
};

}  // namespace host_context
}  // namespace cinn
