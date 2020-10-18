#pragma once
#include <glog/logging.h>
#include <utility>
#include <variant>
#include "cinn/common/object.h"
#include "cinn/common/shared.h"

namespace cinn {
namespace host_context {

using ValueVariantType = std::variant<int32_t, int64_t, float, double, bool>;
class _Value_ : public common::Object {
 public:
  using variant_type = ValueVariantType;

  explicit _Value_(int32_t x) : data(x) {}
  explicit _Value_(int64_t x) : data(x) {}
  explicit _Value_(float x) : data(x) {}
  explicit _Value_(double x) : data(x) {}
  explicit _Value_(bool x) : data(x) {}

  const char* type_info() const override;

  ValueVariantType data;

 private:
  static constexpr const char* __type_info__ = "host_context_value";
};

/**
 * Represents any value types in host context.
 */
class Value : common::Shared<_Value_> {
 public:
  Value() = default;
  explicit Value(_Value_* n) : common::Shared<_Value_>(n) {}
  explicit Value(int32_t val);
  explicit Value(int64_t val);
  explicit Value(float val);
  explicit Value(double val);
  explicit Value(bool val);

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
      p_ = common::make_shared<_Value_>();
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
