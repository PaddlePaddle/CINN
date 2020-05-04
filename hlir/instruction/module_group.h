#pragma once

#include <string>
#include <vector>

#include "hlir/instruction/module.h"

namespace hlir {
namespace instruction {

template <typename T>
struct iterator {
  using value_type = T;
  using self_type  = iterator;
  using reference  = value_type&;
  using pointer    = value_type*;

  explicit iterator(T* data) : data_(data) {}
  self_type operator++() {
    self_type i = *this;
    data_++;
    return i;
  }
  reference operator*() { return *data_; }
  pointer operator->() { return data_; }
  bool operator==(const self_type& other) const { return data_ == other.data_; }
  bool operator!=(const self_type& other) const { return data_ != other.data_; }

 private:
  pointer data_;
};

template <typename T>
struct const_iterator {
  using value_type = const T;
  using self_type  = const_iterator;
  using reference  = value_type&;
  using pointer    = value_type*;

  explicit const_iterator(const T* data) : data_(data) {}
  self_type operator++() {
    self_type i = *this;
    data_++;
    return i;
  }
  reference operator*() const { return *data_; }
  pointer operator->() const { return data_; }
  bool operator==(const self_type& other) const { return data_ == other.data_; }
  bool operator!=(const self_type& other) const { return data_ != other.data_; }

 private:
  const pointer data_;
};

/**
 * Represent a set of Modules those are executed in multiple threads.
 */
class ModuleGroup {
 public:
  using iterator_t       = iterator<Module*>;
  using const_iterator_t = const_iterator<Module*>;

  explicit ModuleGroup(const std::vector<Module*>& group) : group_(group) {}

  iterator_t begin() { return iterator_t(group_.data()); }
  iterator_t end() { return iterator_t(group_.data() + group_.size()); }

  const_iterator_t begin() const { return const_iterator_t(group_.data()); }
  const_iterator_t end() const { return const_iterator_t(group_.data() + group_.size()); }

 private:
  std::vector<Module*> group_;
};

}  // namespace instruction
}  // namespace hlir
