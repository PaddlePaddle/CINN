#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

class FunctionBase : public IrNode {
public:
  virtual const std::string& func_name() const = 0;
};

class FunctionRef : public IrNodeRef {
 public:
  FunctionRef() = default;
  FunctionRef(IrNode* n) : IrNodeRef(n) {}
};

}  // namespace ir
}  // namespace cinn
