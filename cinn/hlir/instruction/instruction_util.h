#pragma once
/*
 * This file defines some helper functions to make HLIR usage easier.
 */

#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace hlir {
namespace instruction {

Instruction* Add(Instruction* a, Instruction* b);
Instruction* Sub(Instruction* a, Instruction* b);
Instruction* Mul(Instruction* a, Instruction* b);
Instruction* Div(Instruction* a, Instruction* b);

Instruction* Tanh(Instruction* x);
Instruction* Ceil(Instruction* x);
Instruction* Abs(Instruction* x);

Instruction* Dot(Instruction* a, Instruction* b);

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
