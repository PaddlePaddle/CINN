#pragma once

#include <glog/logging.h>
#include <llvm/ADT/ArrayRef.h>
#include <utility>
#include "cinn/host_context/value.h"

namespace cinn {
namespace host_context {

template <typename T>
class Argument {
 public:
  explicit Argument(Value value) : value_(value) {}

  Value& value() { return value_; }
  const Value& value() const { return value_; }

  T& get() const { return value_.get<T>(); }

 private:
  Value value_;
};

class RemainingArguments {
 public:
  explicit RemainingArguments(llvm::ArrayRef<Value> remaining_arguments) : remaining_arguments_(remaining_arguments) {}

  llvm::ArrayRef<Value> values() const { return remaining_arguments_; }
  size_t size() const { return remaining_arguments_.size(); }
  const Value& operator[](size_t i) const { return remaining_arguments_[i]; }

 private:
  llvm::ArrayRef<Value> remaining_arguments_;
};

template <typename T>
class Result {
 public:
  explicit Result(Value* result) : result_(result) {}

  template <typename... Args>
  void Emplace(Args&&... args) {
    Value v;
    Set(T(std::forward<Args>(args)...));
  }

  void Set(Argument<T> argument) {
    CHECK(!result_->IsValid());
    *result_ = argument.value();
  }

 private:
  Value* result_{};
};

}  // namespace host_context
}  // namespace cinn
