#pragma once
#include "cinn/ir/ir.h"
#include "cinn/ir/module.h"

namespace cinn {
namespace optim {

/**
 * Optimize the expression but Module.
 * @param e
 * @param runtime_debug_info
 * @return
 */
Expr Optimize(Expr e, Target target, bool runtime_debug_info = false);

/**
 * Optimize a Module.
 */
ir::Module Optimize(const ir::Module& module, const Target& target);

}  // namespace optim
}  // namespace cinn
