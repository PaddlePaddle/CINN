#include "cinn/utils/string.h"

#include <stdarg.h>

#include <cstring>

namespace cinn {
namespace utils {

std::string StringFormat(const std::string fmt_str, ...) {
  /* Reserve two times as much as the length of the fmt_str */
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);                 /* Wrap the plain char array into the unique_ptr */
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

std::string Join(const std::vector<std::string> &fields, const std::string &splitter) {
  std::stringstream ss;
  if (fields.empty()) return "";
  for (int i = 0; i < fields.size() - 1; i++) {
    ss << fields[i];
    ss << splitter;
  }

  if (fields.size() > 1) {
    ss << fields.back();
  }

  return ss.str();
}

std::string Trim(const std::string &s, const char *empty) {
  if (s.empty()) return s;
  auto start = s.find_first_not_of(empty);
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(empty);
  return s.substr(start, end - start + 1);
}

std::string Uppercase(const std::string &x) {
  auto res = x;
  for (auto &c : res) {
    c = toupper(c);
  }
  return res;
}

bool Startswith(const std::string &x, const std::string &str) { return x.find(str) == 0; }
bool Endswith(const std::string &x, const std::string &str) { return x.rfind(str) == x.size() - str.size(); }

}  // namespace utils
}  // namespace cinn
