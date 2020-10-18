#pragma once

#include <glog/logging.h>
#include <llvm/ADT/ArrayRef.h>

#include <utility>

#include "cinn/host_context/kernel_frame.h"
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

template <typename F, F f>
struct KernelImpl;

template <typename T>
struct TypeTag {};

#define CINN_KERNEL(...) ::cinn::host_context::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelImpl<Return (*)(Args...), impl_fn> {
  static void Invoke(KernelFrame* frame) { KernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0, 0>(frame); }

  // Helper that introspects the arguments to derive the signature and cast
  // parts of the KernelFrame to their type before passing them to impl_fn.
  template <typename... RemainingArgs>
  struct KernelCallHelper;

  // Casts the return value of the kernel, if non-void.
  // bool _ is an unnecessary parameter to make compiler allow templace specific in non-namespace scope.
  template <typename T, bool _>
  struct KernelReturnHelper {
    static void Invoke(KernelFrame* frame, const Args&... args) {
      LOG(INFO) << "Handle return: ";
      HandleReturn(frame, impl_fn(args...));
    }
  };

  template <bool _>
  struct KernelReturnHelper<void, _> {
    static void Invoke(KernelFrame* frame, const Args&... args) { impl_fn(args...); }
  };

  // Specialization to cast a single input argument(Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Argument<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1, "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0, "Arguments and results should appear before attributes.");

      Argument<Head> arg(frame->GetArgAt(in_idx));
      KernelCallHelper<Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame, pargs..., arg);
    }
  };

  // Specialization to cast a single result argument (Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Result<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(out_idx != -1, "Do not place Results after RemainingResults");
      static_assert(const_idx == 0, "Arguments and results should appear before attributes");
      Result<Head> arg(&frame->GetResults()[out_idx]);
      KernelCallHelper<Tail...>::template Invoke<in_idx, out_idx + 1, const_idx>(frame, pargs..., arg);
    }
  };

  // Treat other pointer as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head*, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1, "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0, "Arguments and results should appear before attributes.");
      auto* arg = &frame->GetArgAt<Head>(in_idx);
      KernelCallHelper<Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame, pargs..., arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head, Tail...> {
    using ArgT = std::decay_t<Head>;

    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1, "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0, "Arguments and results should appear before attributes.");

      auto& value = frame->GetArgAt(in_idx);
      auto&& arg  = value.get<ArgT>();

      KernelCallHelper<Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame, pargs..., arg);
    }
  };

  // No arguments left.
  template <typename T>
  struct KernelCallHelper<TypeTag<T>> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      KernelReturnHelper<Return, false>::Invoke(frame, pargs...);
    }
  };

  // Store the function result back to the output Value in KernelFrame.
  template <typename T>
  static void HandleReturn(KernelFrame* frame, T&& t) {
    assert(frame->GetNumResults() == 1 && "Extra results passed to kernel.");
    StoreResultAt(frame, 0, std::forward<T>(t));
  }

  // Store result as an Value output in KernelFrame.
  template <typename T>
  static void StoreResultAt(KernelFrame* frame, int index, T&& t) {
    frame->EmplaceResult<std::decay_t<T>>(index, std::forward<T>(t));
  }
};

}  // namespace host_context
}  // namespace cinn
