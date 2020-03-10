#pragma once
#include "cinn/backends/codegen_c.h"
#include "cinn/common/common.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"

namespace cinn {

using backends::CodeGenC;
using backends::Outputs;
using ir::Var;
using lang::Buffer;
using lang::Compute;
using lang::Lower;
using lang::Module;
using lang::Placeholder;
using lang::Sum;

}  // namespace cinn
