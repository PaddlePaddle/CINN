#pragma once

#include <memory>
#include <unordered_map>

#include "cinnrt/host_context/value.h"

namespace cinn {
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

  /**
   * Register a state and set value.
   */
  template <typename T>
  Value* Register(std::string_view key, T&& v);

  /**
   * Get a state called \p key.
   */
  Value* Get(std::string_view key) const;

  ~SymbolTable();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace host_context
}  // namespace cinn
