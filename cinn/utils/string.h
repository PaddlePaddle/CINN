#pragma once
#include <sstream>
#include <string>

namespace cinn {
namespace utils {

//! Get the content of a stream.
template <typename T>
std::string GetStreamCnt(const T& x) {
  std::stringstream os;
  os << x;
  return os.str();
}

}  // namespace utils
}  // namespace cinn
