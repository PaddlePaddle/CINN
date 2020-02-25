#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cinn {
namespace ir {

struct Var;

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace common {

//! Get the predifined axis name.
const std::string& axis_name(int level);

//! Generate `naxis` axis using the global names (i,j,k...).
std::vector<ir::Var> GenDefaultAxis(int naxis);

}  // namespace common
}  // namespace cinn
