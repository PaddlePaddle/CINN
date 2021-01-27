#pragma once

#include <memory>
#include <unordered_map>

#include "cinnrt/host_context/value.h"

namespace cinnrt {
namespace host_context {

/**
 * SymbolTable holds all the states of the kernel graph in the runtime.
 */
class SymbolTable {
 public:
  SymbolTable();

  /**
   * Register a state called \p key.
   */
  Value* Register(std::string_view key);

  Value* Register(std::string_view key, ValueRef value);

  /**
   * Register a state and set value.
   */
  template <typename T>
  Value* Register(std::string_view key, T&& v);

  size_t size() const;

  /**
   * Get a state called \p key.
   */
  Value* GetValue(std::string_view key) const;

  template <typename T>
  T Get(std::string_view key);

  ~SymbolTable();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace host_context
}  // namespace cinnrt
