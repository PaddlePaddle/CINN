#pragma once

#include <map>
#include "cinn/cinn.h"
#include "hlir/instruction/instruction.h"

namespace hlir {
namespace instruction {

/**
 * Variable scope.
 */
class Scope {
 public:
  explicit Scope(Scope* parent = nullptr) : parent_(parent) {}

  cinn::Expr Lookup(const Instruction* instruction) const;

  void Insert(const Instruction* instruction, cinn::Expr expr);

 private:
  std::map<const Instruction*, cinn::Expr> symbol_table_;
  Scope* parent_{};
};

}  // namespace instruction
}  // namespace hlir
