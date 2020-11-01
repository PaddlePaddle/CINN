/**
 * This file exposes some internal APIs to global cinn namespace to make usage more friendly.
 */
#pragma once
#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/common/common.h"
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
using ir::Module;
using ir::Var;
using lang::Buffer;
using lang::CallExtern;
using lang::CallLowered;
using lang::Compute;
using lang::Lower;
using lang::Placeholder;
using lang::ReduceMax;
using lang::ReduceMin;
using lang::ReduceMul;
using lang::ReduceSum;
using optim::Optimize;
using poly::CreateStages;

using lang::logic_and;
using lang::logic_or;

using common::Target;

}  // namespace cinn
