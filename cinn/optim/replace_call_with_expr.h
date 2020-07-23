#pragma once
#include <map>
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace a Call node with a Expr (inline).
 * @param e The expression to modify.
 * @param statement The map from tuple_name to the expression candidate.
 * @param candidate Var of each axis in the expression candidate.
 */
void ReplaceCallWithExpr(Expr *e, const std::string &statement, const Expr &candidate);

/**
 * Replace a Call node with a Expr (inline).
 * @param e The expression to modify.
 * @param statement The map from tuple_name to the expression candidate.
 * @param candidate Var of each axis in the expression candidate.
 * @param axis_map The map from a variable to expression.
 */
void ReplaceIslCallWithExpr(Expr *e,
                            const std::string &statement,
                            const Expr &candidate,
                            const std::map<std::string, Expr> &axis_map);

}  // namespace optim
}  // namespace cinn
