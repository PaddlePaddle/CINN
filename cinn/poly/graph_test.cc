#include "cinn/poly/graph.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/buffer.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::Halide);
  return expr;
}

Stage* CreateStage(const std::string& name, std::vector<Expr>& args, isl::set domain) {
  auto expr = CreateCall(name, args);
  return Stage::New(domain, expr).get();
}

}  // namespace poly
}  // namespace cinn
