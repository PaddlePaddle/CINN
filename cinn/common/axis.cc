#include "cinn/common/axis.h"

#include "cinn/common/common.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/dim.h"
#include "cinn/poly/domain.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace common {

std::vector<ir::Var> GenDefaultAxis(int naxis) {
  std::vector<ir::Var> axis;
  for (int i = 0; i < naxis; i++) {
    axis.emplace_back(common::axis_name(i));
    CHECK(axis.back()->type().valid());
  }
  return axis;
}

std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis) {
  auto vars = GenDefaultAxis(naxis);
  std::vector<Expr> res;
  for (auto& v : vars) {
    res.push_back(Expr(v));
  }
  return res;
}

static std::set<std::string> axis_set() {
  static std::set<std::string> x(kAxises.begin(), kAxises.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) { return axis_set().count(x); }

}  // namespace common
}  // namespace cinn
