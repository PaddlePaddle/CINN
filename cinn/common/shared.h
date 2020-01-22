#pragma once
#include <atomic>
#include <type_traits>

namespace cinn {
namespace common {

class RefCount {
  std::atomic<uint32_t> count{0};

 public:
  using value_type = uint32_t;
  RefCount()       = default;

  value_type Inc() { return ++count; }
  value_type Dec() { return --count; }
  bool is_zero() const { return 0 == count; }
};

class Object;
/**
 * The templated methods are used to unify the way to get the RefCount instance in client classes.
 */
template <typename T>
RefCount& ref_count(const T* t) {
  static_assert(std::is_base_of<Object, T>::value, "T is not a Object");
  return t->__ref_count__;
}
template <typename T>
void Destroy(const T* t) {
  delete t;
}

template <typename T>
struct Shared {
  Shared() = default;
  Shared(T* p) : p_(p) {}
  Shared(const Shared& other) : p_(other.p_) {}
  Shared(Shared&& other) : p_(other.p_) { other.p_ = nullptr; }
  Shared<T>& operator=(const Shared<T>& other) {
    if (other.p_ == p_) return *this;
    // Other can be inside of something owned by this, so we should be careful to incref other before we decref
    // ourselves.
    T* tmp = other.p_;
    IncRef(tmp);
    DesRef(p_);
    p_ = tmp;
    return *this;
  }

  //! Access the pointer in various ways.
  // @{
  T* get() const { return p_; }
  T& operator*() const { return *p_; }
  T* operator->() const { return p_; }
  // @}

  bool defined() const { return p_; }
  bool operator<(const Shared& other) const { return p_ < other.p_; }

  ~Shared() { DesRef(p_); }

 private:
  void IncRef(T* p) {
    if (p) {
      ref_count(p).Inc();
    }
  }

  void DesRef(T* p) {
    if (p) {
      if (ref_count(p).Dec() == 0) {
        Destroy(p);
      }
    }
  }

 protected:
  T* p_{};
};

template <typename T, typename... Args>
T* make_shared(Args... args) {
  return new T(args...);
}

}  // namespace common
}  // namespace cinn
