#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cinn {
namespace utils {

//! Get the content of a stream.
template <typename T>
std::string GetStreamCnt(const T& x);

/**
 * Construct a formatted string with arguments.
 * @param fmt_str The format.
 * @param ... The parameters of the format.
 * @return The formated string.
 */
std::string StringFormat(const std::string& fmt_str, ...);

/**
 * Join multiple fields to a single string. Similar to Python's str.join method.
 */
template <typename T = std::string>
std::string Join(const std::vector<T>& fields, const std::string& splitter) {
  if (fields.empty()) return "";
  std::stringstream ss;
  for (int i = 0; i < fields.size() - 1; i++) ss << fields[i] << splitter;
  ss << fields.back();
  return ss.str();
}

std::vector<std::string> Split(const std::string& str, const std::string& splitter);

std::string Trim(const std::string& s, const char* empty = " \n\r\t");

//! Convert a string to its uppercase.
std::string Uppercase(const std::string& x);

//! Replace a substr 'from' to 'to' in string s.
void Replace(std::string* s, const std::string& from, const std::string& to);

//! Tell if a string \p x start with \p str.
bool Startswith(const std::string& x, const std::string& str);

//! Tell if a string \p x ends with \p str.
bool Endswith(const std::string& x, const std::string& str);

template <typename T>
std::string GetStreamCnt(const T& x) {
  std::stringstream os;
  os << x;
  return os.str();
}

std::string TransValidVarName(std::string name);

bool IsVarNameValid(const std::string& name);

}  // namespace utils
}  // namespace cinn
