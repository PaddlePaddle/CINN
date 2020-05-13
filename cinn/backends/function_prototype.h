#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace backends {

struct FunctionProto {
  using shape_inference_t =
      std::function<std::vector<Expr> /*shape*/ (const std::vector<Expr>& /*arguments*/, int /*value_offset*/)>;

  FunctionProto(const std::string& name,
                const std::vector<Type>& readonly_arg_types,
                const std::vector<Type>& mutable_arg_types,
                Type ret_type                     = Void(),
                shape_inference_t shape_inference = shape_inference_t());

  std::string name;
  std::vector<Type> readonly_arg_types;
  std::vector<Type> mutable_arg_types;
  Type ret_type;

  // Inference the output's shape.
  shape_inference_t shape_inference;

  /**
   * Tell whether the Call \p op matches the function prototype.
   */
  bool Match(const ir::Call* op) const;

  /**
   * Assert the call should match the function prototype.
   */
  void AssertMatch(const ir::Call* op) const;

  /**
   * All the outputs use the n-th argument's shape.
   */
  static shape_inference_t ShapeFollowNthArgument(int n);

 protected:
  void CheckValid();
};

class FunctionProtoRegistry {
 public:
  FunctionProto* Register(std::string_view name, FunctionProto* x);

  FunctionProto* Lookup(std::string_view name);

 private:
  std::unordered_map<std::string_view, std::unique_ptr<FunctionProto>> data_;
};

}  // namespace backends
}  // namespace cinn
