#pragma once

#include <map>

#include "cinn/ir/ir.h"

namespace cinn {
namespace backends {

// borrowed from Halide and TVM.
struct ModularEntry {
  int base;
  int coeff;

  ModularEntry() = default;
  ModularEntry(int base, int coeff) : base(base), coeff(coeff) {}

  static ModularEntry everything() { return ModularEntry{0, 1}; }

  static ModularEntry Add(const ModularEntry& a, const ModularEntry& b);
};

ModularEntry EvalModular(const Expr& e, const std::map<Var, ModularEntry>& mod_map);

}  // namespace backends
}  // namespace cinn
