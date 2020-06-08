#include "cinn/hlir/instruction/primitive/parameter.h"

REGISTER_INSTRUCTION_LOWER(base, Parameter, cinn::hlir::instruction::primitive::ParameterLowerImpl)
