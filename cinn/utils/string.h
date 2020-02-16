#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cinn {
namespace utils {

//! Get the content of a stream.
template <typename T>
std::string GetStreamCnt(const T& x) {
  std::stringstream os;
  os << x;
  return os.str();
}

/**
 * Construct a formatted string with arguments.
 * @param fmt_str The format.
 * @param ... The parameters of the format.
 * @return The formated string.
 */
std::string StringFormat(const std::string fmt_str, ...);

/**
 * Join multiple fields to a single string. Similar to Python's str.join method.
 * @param fields
 * @param splitter
 * @return
 */
std::string Join(const std::vector<std::string>& fields, const std::string& splitter);

std::string Trim(const std::string& s, const char* empty = " \n\r\t");

}  // namespace utils
}  // namespace cinn
