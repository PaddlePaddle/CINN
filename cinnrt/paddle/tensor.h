#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "cinnrt/common/buffer.h"
#include "cinnrt/common/common.h"
#include "cinnrt/common/object.h"

namespace cinnrt {
namespace paddle {
using common::Target;

struct Shape {
  using dim_t = int;

  Shape() = default;
  explicit Shape(const std::vector<dim_t>& data) : data_(data) {}

  void SetData(const std::vector<dim_t>& data) { data_ = data; }

  const std::vector<dim_t>& data() const CINN_RESULT_SHOULD_USE { return data_; }
  std::vector<dim_t>& data() CINN_RESULT_SHOULD_USE { return data_; }
  size_t size() const CINN_RESULT_SHOULD_USE { return data_.size(); }
  uint32_t numel() const CINN_RESULT_SHOULD_USE {
    return std::accumulate(data_.begin(), data_.end(), 1, [](dim_t a, dim_t b) { return a * b; });
  }

 private:
  std::vector<dim_t> data_;
};

class _Tensor_ : public cinnrt::common::Object {
 public:
  _Tensor_() : buffer_(std::make_shared<Buffer>()) {}

  Shape& shape() { return shape_; }

  void Resize(const Shape& shape) {
    shape_ = shape;
    buffer_->data()->resize(reinterpret_cast<const cinn_dimension_t*>(shape.data().data()), shape.size());
  }

  template <typename T>
  inline T* mutable_data(const Target& target) {
    set_type(type_of<T>());
    if (target == common::DefaultHostTarget()) {
      int alignment = type_of<T>().ElementOf().bits();
      buffer_->ResizeLazy(alignment, shape_.numel() * sizeof(T), target);
    } else {
      buffer_->ResizeLazy(shape_.numel() * sizeof(T), target);
    }
    return reinterpret_cast<T*>(buffer_->data()->memory);
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<T*>(buffer_->data()->memory);
  }

  const Type& type() { return type_; }

  void set_type(Type type) { type_ = type; }
  const Type& type() const { return type_; }

  cinn_buffer_t* buffer() { return buffer_->data(); }

  const char* type_info() const override { return __type_info__; }

 private:
  common::Type type_;
  // A shared ptr to make it easier to share buffer between tensors.
  std::shared_ptr<Buffer> buffer_;
  Shape shape_;

  static constexpr char* __type_info__ = "_frontend_tensor_";
};

class Tensor : public Shared<_Tensor_> {
 public:
  Tensor() : Shared(new _Tensor_) {}
  explicit Tensor(_Tensor_* x) : Shared(x) {}
};

}  // namespace paddle
}  // namespace cinnrt
