#pragma once
#include <string>
#include <vector>

#include "cinn/ir/node.h"

namespace cinn {
namespace hlir {
namespace pe {

/**
 * @brief PE that calculates a matrix multiplication
 *
 * @param A The first input tensor
 * @param B The second input tensor
 * @param trans_a whether A is transposed, default: false
 * @param trans_b whether B is transposed, default: false
 * @param x_num_col_dims The mul pe can take tensors with more than two dimensions as its inputs. If the input $x$ is
 * a tensor with more than two dimensions, $x$ will be flattened into a two-dimensional matrix first. The flattening
 * rule is: the first `num_col_dims` will be flattened to form the first dimension of the final matrix (the height of
 * the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to form the second dimension of the
 * final matrix (the width of the matrix). As a result, height of the flattened matrix is equal to the product of
 * $x$'s first `x_num_col_dims` dimensions' sizes, and width of the flattened matrix is equal to the product of $x$'s
 * last `rank(x) - num_col_dims` dimensions' size. For example, suppose $x$ is a 6-dimensional tensor with the shape
 * [2, 3, 4, 5, 6], and `x_num_col_dims` = 3. Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24,
 * 30]. Default is 1
 * @param y_num_col_dims The mul_op can take tensors with more than two dimensions as its inputs. If the input $y$ is a
 *tensor with more than two dimensions, $y$ will be flattened into a two-dimensional matrix first. The attribute
 *`y_num_col_dims` determines how $y$ is flattened. See comments of `x_num_col_dims` for more details. Default is 1.
 * @param name The name of the operation
 *
 * @return the output tensor
 */
ir::Tensor Matmul(const ir::Tensor& A,
                  const ir::Tensor& B,
                  bool trans_a,
                  bool trans_b,
                  int x_num_col_dims,
                  int y_num_col_dims,
                  const std::string& name);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
