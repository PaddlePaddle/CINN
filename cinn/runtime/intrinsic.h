#pragma once
#include <vector>
#include <string>
#include <cinn/ir/buffer.h>

namespace cinn {

namespace ir {
class Expr;
}  // namespace ir

namespace runtime {

static const char* buffer_load = "cinn_buffer_load";

/**
 * Load an element from a buffer.
 * @param buffer
 * @param shape
 * @param indices
 * @return
 */
ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr> &indices);

}  // namespace runtime
}  // namespace cinn
