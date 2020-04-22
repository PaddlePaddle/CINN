#include "hlir/instruction/shape.h"

namespace hlir {
namespace instruction {

bool Shape::operator==(const Shape &other) const {
  if (dims_.size() != other.dims_.size()) return false;
  for (int i = 0; i < num_dims(); i++) {
    if (dims_[i] != other.dims_[i]) return false;
  }
  return true;
}

int Shape::operator[](int offset) const {
  CHECK_LT(offset, dims_.size());
  return dims_[offset];
}

bool Shape::operator!=(const Shape &other) { return !(*this == other); }

int &Shape::operator[](int offset) {
  CHECK_LT(offset, num_dims());
  return dims_[offset];
}

Shape::Shape(std::initializer_list<int> list) {
  for (int x : list) AddDim(x);
}

std::string Shape::to_debug_string() const {
  std::stringstream ss;

  ss << "[";
  if (!dims_.empty()) {
    for (int i = 0; i < dims_.size() - 1; i++) {
      ss << dims_[i] << ", ";
    }
    ss << dims_.back();
  }
  ss << "]";
  return ss.str();
}

}  // namespace instruction
}  // namespace hlir
