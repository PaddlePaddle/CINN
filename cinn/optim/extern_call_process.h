#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void ExternCallMultiOutputShallowStore(Expr* e);

void ExternCallRemoveTupleGetStatements(Expr* e);

}  // namespace optim
}  // namespace cinn
