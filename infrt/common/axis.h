#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace infrt {
namespace common {

//! Get the predefined axis name.
const std::string& axis_name(int level);
bool IsAxisNameReserved(const std::string& x);

}  // namespace common
}  // namespace infrt
