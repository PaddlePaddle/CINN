#pragma once

#include <memory>

#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

struct Specialization {
  Expr condition;
};

struct DefinitionContents;
struct FunctionContents;

/**
 * A Function definition which can either represent a init or an update definition.
 */
class Definition {
 public:
  explicit Definition(const std::shared_ptr<DefinitionContents>& contents) : contents_(contents) {}

 private:
  std::shared_ptr<DefinitionContents> contents_;
};

}  // namespace ir
}  // namespace cinn
