#pragma once
#include <glog/logging.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cinn {
namespace ir {

struct Var;
struct Expr;

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace common {

const static std::vector<std::string> kAxises({
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

//! Get the predifined axis name.
inline const std::string& axis_name(int level) {
  CHECK_LT(level, kAxises.size());
  return kAxises[level];
}

//! Generate `naxis` axis using the global names (i,j,k...).
std::vector<ir::Var> GenDefaultAxis(int naxis);
std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis);

bool IsAxisNameReserved(const std::string& x);

}  // namespace common
}  // namespace cinn
