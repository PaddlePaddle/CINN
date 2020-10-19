/**
 * Lower lowerise the statements to LoweredFuncs.
 */

#pragma once
#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/module.h"
#include "cinn/lang/packed_func.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {
using ir::Tensor;
using poly::StageMap;

/**
 * \brief Lower the computation of \p tensor_args and \p scalar_args to a LoweredFunc.
 * @param name The name of the function.
 * @param tensor_args The tensor arguments, where the computation logic locates.
 * @param scalar_args The scalar arguments, indicate some dimensions.
 * @param temp_tensors The temporary tensors(buffers) used in the body.
 * @param b The module this function belongs to.
 * @return A LoweredFunc, whose name is \p name, the argument list is the concatenation of \p tensor_args and \p
 * scalar_args.
 */
ir::LoweredFunc Lower(const std::string &name,
                      StageMap stages,
                      const std::vector<Tensor> &tensor_args,
                      const std::vector<Var> &scalar_args     = {},
                      const std::vector<Tensor> &temp_tensors = {},
                      Module::Builder *b                      = nullptr,
                      const Target &target                    = common::DefaultHostTarget());

}  // namespace lang
}  // namespace cinn
