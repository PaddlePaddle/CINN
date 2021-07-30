#include "cinnrt/common/axis.h"

#include <glog/logging.h>

#include "cinnrt/common/common.h"

namespace cinnrt {
namespace common {

// std::vector<ir::Var> GenDefaultAxis(int naxis) {
//  std::vector<ir::Var> axis;
//  for (int i = 0; i < naxis; i++) {
//    axis.emplace_back(common::axis_name(i));
//    CHECK(axis.back()->type().valid());
//  }
//  return axis;
//}
//
// std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis) {
//  auto vars = GenDefaultAxis(naxis);
//  std::vector<Expr> res;
//  for (auto& v : vars) {
//    res.push_back(Expr(v));
//  }
//  return res;
//}

const std::vector<std::string> kAxises({
    "i",  // level 0
    "j",  // level 1
    "k",  // level 2
    "a",  // level 3
    "b",  // level 4
    "c",  // level 5
    "d",  // level 6
    "e",  // level 7
    "f",  // level 8
    "g",  // level 9
    "h"   // level 10
});

static std::set<std::string> axis_set() {
  static std::set<std::string> x(kAxises.begin(), kAxises.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) { return axis_set().count(x); }

const std::string& axis_name(int level) {
  CHECK_LT(level, kAxises.size());
  return kAxises[level];
}

}  // namespace common
}  // namespace cinnrt
