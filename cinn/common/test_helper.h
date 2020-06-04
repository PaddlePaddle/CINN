#pragma once

#include <string>
#include <vector>

#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/cinn.h"
#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace common {

/**
 * Create buffer for test.
 *
 * usage:
 *
 * auto* buf = BufferBuilder(Float(32), {20, 20}).set_random().Build();
 */
struct BufferBuilder {
  explicit BufferBuilder(Type type, const std::vector<int>& shape) : type_(type), shape_(shape) {}

  BufferBuilder& set_random() {
    init_type_ = 1;
    return *this;
  }

  BufferBuilder& set_zero() {
    init_type_ = 0;
    return *this;
  }

  BufferBuilder& set_align(int align) {
    align_ = align;
    return *this;
  }

  cinn_buffer_t* Build();

 private:
  template <typename T>
  void RandomFloat(void* arr, int len) {
    auto* data = static_cast<T*>(arr);
    for (int i = 0; i < len; i++) {
      data[i] = static_cast<T>(rand()) / RAND_MAX;  // NOLINT
    }
  }

  template <typename T>
  void RandomInt(void* arr, int len) {
    auto* data = static_cast<T*>(arr);
    for (int i = 0; i < len; i++) {
      data[i] = static_cast<T>(rand() % std::numeric_limits<T>::max());  // NOLINT
    }
  }

 private:
  std::vector<int> shape_;
  int init_type_ = 0;  // 0 for zero, 1 for random
  int align_{};
  Type type_;
};

struct ArgsBuilder {
  template <typename T>
  ArgsBuilder& Add(T x) {
    data_.emplace_back(x);
    return *this;
  }

  std::vector<cinn_pod_value_t> Build() {
    CHECK(!data_.empty());
    return data_;
  }

 private:
  std::vector<cinn_pod_value_t> data_;
};

}  // namespace common
}  // namespace cinn
