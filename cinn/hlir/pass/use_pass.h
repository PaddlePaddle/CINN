#pragma once

#include "cinn/common/macros.h"

CINN_USE_REGISTER(InferShape)
#ifdef CINN_WITH_CUDA
CINN_USE_REGISTER(OpFusion)
#endif