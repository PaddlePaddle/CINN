#pragma once
#include <glog/logging.h>

#include <any>
#include <type_traits>
#include <variant>

#include "cinn/common/cinn_value.h"
#include "cinn/utils/small_vector.h"

namespace cinn::hlir::framework {

class AnyValue : Object {
 public:
  std::any value;

  static constexpr const char* __type_info__ = "AnyValue";

 private:
  const char* type_info() const override;
};

template <typename T>
struct AnyValueRef : Shared<AnyValue> {
  AnyValueRef() : Shared<AnyValue>(new AnyValue) {}
  AnyValueRef(AnyValue* n) : Shared<AnyValue>(n) {}
  AnyValueRef(T x) : Shared<AnyValue>(new AnyValue) { *get() = x; }
};

template <typename T>
class Argument {
 public:
  explicit Argument(AnyValue* value) : value_(value) {}

  AnyValue* value() const { return value_; }
  T& get() const { return *value_; }
  T* operator->() { return &std::get<T&>(*value_); }
  const T* operator->() const { return &std::get<const T&>(*value_); }

 private:
  AnyValue* value_{};
};

struct KernelFrame {
 public:
  int num_arguments() const { return num_arguments_; }
  int num_results() const { return num_results_; }

  template <typename T>
  T& GetArgAt(int idx) const {
    return std::get<T&>(GetArgAt(idx));
  }

  AnyValue* GetArgAt(int idx) const {
    CHECK_LT(idx, num_arguments_);
    return registers_[idx];
  }

  template <typename T, typename... Args>
  void EmplaceResultAt(int index, Args&&... args) {
    CHECK_LT(index, num_results()) << "Invalid result index";
    AnyValue* result = GetResultAt(index);
    CHECK(!result->value.has_value()) << "result value is non-empty";
    result->value.emplace<T>(std::forward<Args>(args)...);
  }

  AnyValue* GetResultAt(int index) const {
    CHECK_LT(index, result_indices_.size());
    return registers_[result_indices_[index]];
  }

 private:
  unsigned num_arguments_;
  unsigned num_results_;
  utils::SmallVector<AnyValue*, 6> registers_;
  //! Indices into `registers_`.
  utils::SmallVector<uint32_t, 6> argument_indices_;
  //! Indices into `registers_`.
  utils::SmallVector<uint32_t, 6> result_indices_;
};

template <typename Head, typename... Tail>
struct KernelCallHelper {
  template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
  static void Invoke(KernelFrame* frame, const PreviousArgs&... pre_args) {
    static_assert(in_idx != -1, "Do not place Arguments after RemainingArgs");
    Argument<Head> arg(frame->GetArgAt(in_idx));
    KernelCallHelper<Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame, pre_args..., arg);
  }
};

// Treat any other type as an Argument.
template <typename Head, typename... Tail>
struct KernelCallHelper<Head, Tail...> {
  using ArgT = std::decay_t<Head>;

  template <typename T>
  static T GetArg(AnyValue* value, std::true_type) {
    return T(&value->template get<typename ArgT::UnderlyingT>());
  }

  template <typename T>
  static T& GetArg(AnyValue* value, std::false_type) {
    return value->get<ArgT>();
  }

  template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
  static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
    static_assert(in_idx != -1, "Do not place Arguments after RemainingArguments");
    ;
    static_assert(out_idx == 0, "Arguments shoud");
  }
};

template <typename F, F f>
struct CinnKernelImpl;

template <typename T>
struct TypeTag {};

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct CinnKernelImpl<Return (*)(Args...), impl_fn> {
  static void Invoke(KernelFrame* frame) { KernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0, 0>(frame); }

 private:
  template <typename... RemainingArgs>
  struct KernelCallHelper;

  template <typename T>
  struct KernelReturnHelper {
    static void Invoke(KernelFrame* frame, const Args&... args) { HandleReturn(frame, impl_fn(args...)); }
  };

  template <typename T>
  static void StoreResultAt(KernelFrame* frame, int index, T&& t) {
    frame->EmplaceResultAt<std::decay_t<T>>(index, std::forward<T>(t));
  }

  // Store the function result back to the output Value in the KernelFrame.
  template <typename T>
  static void HandleReturn(KernelFrame* frame, T&& t) {
    assert(frame->num_results() == 1 && "Extra results passed to kernel.");
    StoreResultAt(frame, 0, std::forward<T>(t));
  }

  template <typename TupleT, size_t... I>
  static void EmplaceTupleResult(KernelFrame* frame, TupleT&& res, std::index_sequence<I...>) {
    std::ignore = std::initializer_list<int>{(StoreResultAt(frame, I, std::get<I>(std::forward<TupleT>(res))), 0)...};
  }

  template <typename... T>
  static void HandleReturn(KernelFrame* frame, std::tuple<T...> t) {
    assert(frame->num_results() == sizeof...(T) && "Incorrect number of results passed to kernel");
    EmplaceTupleResult(frame, std::move(t), std::make_index_sequence<sizeof...(T)>{});
  }
};

#define CINN_KERNEL(...) ::cinn::hlir::framework::CinnKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke

}  // namespace cinn::hlir::framework
