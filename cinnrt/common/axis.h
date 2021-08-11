#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cinnrt {
namespace common {

//! Get the predifined axis name.
const std::string& axis_name(int level);
bool IsAxisNameReserved(const std::string& x);

}  // namespace common
}  // namespace cinnrt
