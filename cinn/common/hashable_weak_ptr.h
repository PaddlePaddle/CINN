#pragma once
#include <functional>
#include <memory>

namespace cinn {
namespace common {
  
template<typename T>
class HashableWeakPtr {
 public:
  HashableWeakPtr(const std::shared_ptr<T>& shared_ptr): ptr_(shared_ptr.get()), weak_ptr_(shared_ptr) {}
  HashableWeakPtr(const std::weak_ptr<T>& weak_ptr): ptr_(weak_ptr.lock().get()), weak_ptr_(weak_ptr) {}
  HashableWeakPtr(const HashableWeakPtr&) = default;
  HashableWeakPtr(HashableWeakPtr&&) = default;
  HashableWeakPtr() : ptr_(), weak_ptr_() {}

  const std::weak_ptr<T>& get() const { return weak_ptr_; }

  std::shared_ptr<T> lock() const { return weak_ptr_.lock(); }

  HashableWeakPtr& operator=(const HashableWeakPtr& other) const {
    ptr_ = other.ptr_;
    weak_ptr_ = other.weak_ptr_;
    return *this;
  }

  HashableWeakPtr& operator=(HashableWeakPtr&& other) {
    ptr_ = std::move(other.ptr_);
    weak_ptr_ = std::move(other.weak_ptr_);
    return *this;
  }

  bool operator==(const HashableWeakPtr& other) const {
    return this->ptr_ == other.ptr_;
  }

  size_t hash_value() const { return reinterpret_cast<size_t>(ptr_); }

 private:
  T* ptr_;
  std::weak_ptr<T> weak_ptr_;
};

}
}

namespace std {

template<typename T>
struct hash<cinn::common::HashableWeakPtr<T>> {
  size_t operator()(const cinn::common::HashableWeakPtr<T>& ptr) const { return ptr.hash_value(); }
};

}

