#include "cinn/runtime/buffer.h"

namespace cinn {
namespace runtime {

Shape::Shape(const Shape &other) : data_(new value_type[other.ndims()]), ndims_(other.ndims()) {
  if (ndims() > 0) {
    memcpy(data_, other.data(), ndims_ * sizeof(value_type));
  }
}

void Shape::Resize(int ndim) {
  CHECK_GT(ndim, 0);
  ndims_ = ndim;
  if (data_) delete data_;
  data_ = new value_type[ndim];
}

Shape::value_type &Shape::operator[](int i) {
  CHECK_GT(ndims_, 0) << "shape is empty";
  CHECK_LT(i, ndims_) << "index " << i << "out of range " << ndims_;
  return data_[i];
}

Shape::value_type Shape::operator[](int i) const {
  CHECK_GT(ndims_, 0) << "shape is empty";
  CHECK_LT(i, ndims_) << "index " << i << "out of range " << ndims_;
  return data_[i];
}

uint32_t Shape::num_elements() const {
  uint32_t res = ndims_ > 0 ? 1 : 0;
  for (int i = 0; i < ndims(); i++) res *= (*this)[i];
  return res;
}

}  // namespace runtime
}  // namespace cinn
