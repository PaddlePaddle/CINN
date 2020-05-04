#pragma once

#include <string>
#include <vector>

#include "cinn/cinn.h"
#include "hlir/instruction/context.h"
#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {
namespace primitive {

using cinn::ir::Tensor;

struct DotImpl {
  explicit DotImpl(Context* ctx) : ctx_(ctx) {}

  Tensor operator()(const Tensor& a, const Tensor& b, const std::string& name);

  /**
   * \brief Vector dot vector.
   * @param a vector [n]
   * @param b vector [n]
   * @param name Name of the output tensor.
   * @return Result tensor.
   */
  Tensor VecDotVec(const Tensor& a, const Tensor& b, const std::string& name);

  /**
   * \brief Matrix dot matrix.
   * @param a Matrix [m x k]
   * @param b Matrix [k x n]
   * @param name Name of the output tensor.
   * @return Result tensor.
   */
  Tensor MatDotMat(const Tensor& a, const Tensor& b, const std::string& name);

  /**
   * \brief Matrix dot vector.
   * @param a Matrix [m x k]
   * @param b Vector [k]
   * @param name Name of the output tensor.
   * @return Result tensor.
   */
  Tensor MatDotVec(const Tensor& a, const Tensor& b, const std::string& name);

 private:
  Context* ctx_{};
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
