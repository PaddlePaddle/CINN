#pragma once
#include <cinn/ir/buffer.h>

#include <string>
#include <vector>

namespace cinn {

namespace ir {
class Expr;
}  // namespace ir

namespace runtime {

//! cinn_buffer_t::new_(buffer)
static const char* buffer_create = "cinn_buffer_t::new_";
//! cinn_buffer_t::delete_(buffer)
static const char* buffer_destroy = "cinn_buffer_t::delete_";

static const char* buffer_load = "cinn_buffer_load";

static const char* buffer_malloc = "cinn_buffer_malloc";

ir::Expr BufferCreate(ir::Buffer buffer);
/**
 * Get an expression to load an element from a buffer.
 * @param buffer
 * @param shape
 * @param indices
 * @return
 */
ir::Expr BufferLoad(ir::Buffer buffer, const std::vector<ir::Expr>& indices);

/**
 * Get an expression to malloc a buffer.
 * @param buffer
 * @return
 */
ir::Expr BufferMalloc(ir::Buffer buffer);

}  // namespace runtime
}  // namespace cinn
