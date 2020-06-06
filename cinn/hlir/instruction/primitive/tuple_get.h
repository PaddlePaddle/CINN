#pragma once
#include <string>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/hlir/instruction/context.h"
#include "cinn/hlir/instruction/instruction.h"
#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/module_lower.h"
#include "cinn/hlir/instruction/shape.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

void LowerTupleGet(const Instruction* instr, ComputationLower* lower) {
  CHECK(instr);
  CHECK(lower);

  auto* tuple_get = instr->As<TupleGet>();
  if (tuple_get->tuple()->call()) {
    const std::vector<Expr>& call_ret = lower->get_call_ret(tuple_get->tuple()->call());
    lower->scope().Insert(instr, call_ret[tuple_get->offset()]);
  } else if (!tuple_get->tuple()->items().empty()) {
    auto* key = tuple_get->tuple()->items()[tuple_get->offset()];
    auto expr = lower->scope().Lookup(key);
    CHECK(expr.defined());
    lower->scope().Insert(instr, expr);
  } else {
    NOT_IMPLEMENTED
  }
}

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
