#pragma once

#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * \brief Rewrite the Call Nodes marked type as CINN, pack their arguments into `void*, int` so that they can trigger a
 * `LoweredFunc`.
 *
 * For example, input the IR
 * \code
 * Call(some_lowered_func, a:cinn_buffer_t*, b:cinn_buffer_t*, c:cinn_buffer_t*)
 * \endcode
 *
 * This pass will rewrite it to
 * \code
 * cinn_pod_value_t a_(a);
 * cinn_pod_value_t b_(b);
 * cinn_pod_value_t c_(c);
 *
 * cinn_args_construct(packed_args, a_, b_, c_);
 * Call(some_lowered_func, packed_args, 3); // 3 is the number of arguments
 * \endcode
 */
void FoldCINNCallArguments(Expr* expr);

}  // namespace optim
}  // namespace cinn
