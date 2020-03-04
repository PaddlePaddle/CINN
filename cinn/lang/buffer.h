#pragma once

#include "cinn/ir/buffer.h"

namespace cinn {
namespace lang {

/**
 * This is a DSL wrapper for ir::Buffer.
 */
class Buffer {
 public:
  explicit Buffer(Type type, const std::string& name = "");

  ir::_Buffer_* operator->() { return buffer_->As<ir::_Buffer_>(); }
  const ir::_Buffer_* operator->() const { return buffer_->As<ir::_Buffer_>(); }

  ir::Buffer buffer() const { return buffer_; }

 private:
  ir::Buffer buffer_;
};

}  // namespace lang
}  // namespace cinn
