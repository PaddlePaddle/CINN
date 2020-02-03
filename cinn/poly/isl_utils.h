#pragma once

#include <isl/cpp.h>
#include <string>
#include <vector>

namespace cinn {
namespace poly {

//! Get dimension names from isl containers.
std::vector<std::string> GetDimNames(const isl::set &x);

}  // namespace poly
}  // namespace cinn
