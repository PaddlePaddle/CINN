#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

/**
 * Collect the tensors(without duplication) in the expression.
 */
std::set<Expr> CollectIRNodes(Expr x, std::function<bool(const Expr*)>&& teller);

/**
 * Collect the tensors in Load nodes.
 */
std::set<Expr> CollectLoadTensors(Expr x, std::function<bool(const Expr*)>&& teller);

/**
 * Collect the tensors in Store nodes.
 */
std::set<Expr> CollectStoreTensors(Expr x, std::function<bool(const Expr*)>&& teller);

std::set<Expr> CollectReferencedTensors(Expr x, const std::function<bool(const Expr*)>& teller);

std::map<std::string, Expr> CollectTensorMap(
    Expr x, std::function<bool(const Expr*)>&& extra_teller = [](const Expr* x) { return true; });

}  // namespace ir
}  // namespace cinn
