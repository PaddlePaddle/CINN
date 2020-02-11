#pragma once
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace ir {

struct IrVisitor {
#define __m(t__) virtual void Visit(const t__* x) = 0;

  NODETY_FORALL(__m)

#undef __m
};

}  // namespace ir
}  // namespace cinn
