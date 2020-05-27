#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

ir::Tensor Pad(ir::Tensor t,
               const std::vector<Expr>& pad_before,
               std::vector<Expr> pad_after = {},
               Expr pad_value              = Expr(),
               const std::string& name     = "pad_",
               const std::string& pad_mode = "constant");

ir::Tensor Conv2dNCHW(ir::Tensor I,
                      ir::Tensor W,
                      int pad_h               = 0,
                      int pad_w               = 0,
                      int stride_h            = 1,
                      int stride_w            = 1,
                      const std::string& name = "conv2d_nchw");

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
