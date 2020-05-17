#include "cinn/hlir/instruction/scope.h"

namespace cinn {
namespace hlir {
namespace instruction {

cinn::Expr Scope::Lookup(const Instruction *instruction) const {
  // find local
  {
    auto it = symbol_table_.find(instruction);
    if (it != symbol_table_.end()) return it->second;
  }

  // find in parent recursively.

  auto *p = parent_;
  while (p) {
    auto item = p->Lookup(instruction);
    if (item.defined()) return item;

    p = p->parent_;
  }

  return cinn::Expr();
}

void Scope::Insert(const Instruction *instruction, cinn::Expr expr) {
  CHECK(expr.defined()) << "symbol's expression is undefined";
  CHECK(!symbol_table_.count(instruction)) << "Duplicate symbol in local scope " << instruction->id();
  symbol_table_[instruction] = expr;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
