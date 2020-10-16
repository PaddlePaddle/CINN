#pragma once
#include <glog/logging.h>
#include <llvm/ADT/ArrayRef.h>
#include <utility>

#include "cinn/host_context/value.h"
#include "cinn/utils/small_vector.h"

namespace cinn::host_context {

class KernelFrame {
 public:
  int GetNumArgs() const { return num_arguments_; }
  int GetNumResults() const { return num_results_; }

  template <typename T>
  T& GetArgAt(int index) const {
    CHECK_LT(index, GetNumArgs());
    return value_or_attrs_[index];
  }

  Value& GetArgAt(int index) {
    CHECK_LT(index, GetNumArgs());
    return value_or_attrs_[index];
  }

  template <typename T, typename... Args>
  void EmplaceResult(Args&&... args) {
    EmplaceResult<T>(0, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  void EmplaceResult(int index, Args&&... args) {
    SetResultAt(index, Value(T(std::forward<Args>(args)...)));
  }

  void SetResultAt(int index, Value value) {
    CHECK_LT(index, num_results_) << "Invalid result index";
    value_or_attrs_[index] = value;
  }

  llvm::ArrayRef<Value> GetResults() const { return GetValues(num_arguments_, num_results_); }
  llvm::MutableArrayRef<Value> GetResults() { return GetMutableValues(num_arguments_, num_results_); }

  llvm::ArrayRef<Value> GetValues(size_t from, size_t length) const {
    CHECK_LE(from + length, num_arguments_ + num_results_);
    if (length == 0) return {};

    return llvm::makeArrayRef(&value_or_attrs_[from], length);
  }

  llvm::MutableArrayRef<Value> GetMutableValues(size_t from, size_t length) {
    CHECK_LE(from + length, num_arguments_ + num_results_);
    if (length == 0) return {};
    return llvm::makeMutableArrayRef(&value_or_attrs_[from], length);
  }

 protected:
  int num_arguments_{};
  int num_results_{};

  utils::SmallVector<Value, 8> value_or_attrs_;
};

class KernelFrameBuilder : public KernelFrame {
 public:
  void AddArgument(Value value) {
    CHECK_EQ(num_results_, -1) << "Should call AddArgument before calling SetNumResults";
    value_or_attrs_.push_back(value);
    ++num_arguments_;
  }

  void SetNumResults(size_t n) {
    CHECK_EQ(num_arguments_, value_or_attrs_.size());
    CHECK_EQ(num_results_, -1);
    num_results_ = n;
    value_or_attrs_.resize(value_or_attrs_.size() + n);
  }

  void Reset() {
    value_or_attrs_.clear();
    num_arguments_ = 0;
    num_results_   = -1;
  }
};

}  // namespace cinn::host_context
