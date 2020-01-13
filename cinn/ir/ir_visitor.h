#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

struct IrVisitor {
#define __m(t__) virtual void Visit(t__* x) = 0;

  NODETY_FORALL(__m)

#undef __m
};

}  // namespace ir
}  // namespace cinn
