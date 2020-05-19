#pragma once

#include "cinn/hlir/instruction/shape.h"

namespace cinn {
namespace hlir {
namespace instruction {

struct Instruction;

Shape BinaryInferenceShape(Instruction* a, Instruction* b);

Shape DotInferenceShape(Instruction* a, Instruction* b);

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
