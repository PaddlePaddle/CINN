#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/buffer.h"

namespace cinn {
namespace hlir {
namespace framework {
using common::Target;

struct Shape {
  using dim_t = uint32_t;

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

class Tensor final {
 public:
  Tensor() : buffer_(std::make_shared<Buffer>()) {}

  const Shape& shape() const { return shape_; }

  void Resize(const Shape& shape) {
    shape_ = shape;
    buffer_->data()->resize(reinterpret_cast<const cinn_dimension_t*>(shape.data().data()), shape.size());
  }

  template <typename T>
  inline T* mutable_data(const Target& target) {
    set_type(type_of<T>());
    buffer_->ResizeLazy(shape_.numel() * sizeof(T), target);
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

 private:
  common::Type type_;
  // A shared ptr to make it easier to share buffer between tensors.
  std::shared_ptr<Buffer> buffer_;
  Shape shape_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
