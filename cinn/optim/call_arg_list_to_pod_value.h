#pragma once
/**
 * \file Transform the CINN Call node's args to cinn_pod_value_t array.
 */

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

void CallArgListToPodValue(Expr* e);

}  // namespace optim
}  // namespace cinn
