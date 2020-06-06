#pragma once
#include "cinn/hlir/instruction/lower_impl.h"

USE_INSTRUCTION_LOWER(base, Parameter)

// binarys @{
USE_INSTRUCTION_LOWER(base, Add)
USE_INSTRUCTION_LOWER(base, Sub)
USE_INSTRUCTION_LOWER(base, Mul)
USE_INSTRUCTION_LOWER(base, Div)
// @}

// elementwise @{
USE_INSTRUCTION_LOWER(base, Tanh)
USE_INSTRUCTION_LOWER(base, Ceil)
USE_INSTRUCTION_LOWER(base, Abs)
USE_INSTRUCTION_LOWER(base, Exp)
// @}

USE_INSTRUCTION_LOWER(base, Dot)
USE_INSTRUCTION_LOWER(base, Conv)
USE_INSTRUCTION_LOWER(base, Call)
USE_INSTRUCTION_LOWER(base, Tuple)
USE_INSTRUCTION_LOWER(base, TupleGet)
