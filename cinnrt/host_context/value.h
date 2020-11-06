#pragma once
#include <glog/logging.h>

#include <utility>
#include <variant>

#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinnrt/host_context/dense_tensor.h"
#include "cinnrt/host_context/dense_tensor_view.h"
#include "cinnrt/host_context/tensor_shape.h"
#include "llvm/ADT/SmallVector.h"

namespace cinn {
namespace host_context {

using ValueVariantType = std::variant<int16_t,
                                      int32_t,
                                      int64_t,
                                      float,
                                      double,
                                      bool,
                                      TensorShape,
                                      DenseTensor,
                                      std::vector<int16_t>,
                                      std::vector<int32_t>,
                                      std::vector<int64_t>,
                                      std::vector<float>,
                                      std::vector<double>>;

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
  explicit Value(std::vector<int16_t>&& x) : data(x) {}
  explicit Value(std::vector<int32_t>&& x) : data(x) {}
  explicit Value(std::vector<int64_t>&& x) : data(x) {}
  explicit Value(std::vector<float>&& x) : data(x) {}
  explicit Value(std::vector<double>&& x) : data(x) {}
  explicit Value(TensorShape&& x) : data(std::move(x)) {}
  explicit Value(DenseTensor&& x) : data(std::move(x)) {}

  template <typename T>
  const T& get() const {
    return std::get<T>(data);
  }
  template <typename T>
  T& get() {
    return std::get<T>(data);
  }

  template <typename T>
  void set(T&& v) {
    data = std::move(v);
  }

  const char* type_info() const override;

 private:
  ValueVariantType data;
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
  const T& get() const {
    CHECK(p_);
    return p_->get<T>();
  }

  template <typename T>
  T& get() {
    CHECK(p_);
    return p_->get<T>();
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
