/**
 * This file exposes some internal APIs to global cinn namespace to make usage more friendly.
 */
#pragma once
#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/common/common.h"
#include "cinn/hlir/primitive/add.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/optim/optimize.h"

namespace cinn {

using backends::CodeGenC;
using backends::CodeGenCX86;
using backends::Outputs;
using ir::Var;
using lang::Buffer;
using lang::Compute;
using lang::Lower;
using lang::Module;
using lang::Placeholder;
using lang::Sum;
using optim::Optimize;

using ir::logic_and;
using ir::logic_or;

using common::Target;

}  // namespace cinn
