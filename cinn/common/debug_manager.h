#pragma once
#include <string>
#include <utility>
#include <vector>

#include "cinn/utils/any.h"

namespace cinn {
namespace common {

/**
 * Container for debug info.
 * DebugManager is integrated into the global Context, and used to log something(but not print to stdout directly).
 */
class DebugManager {
 public:
  void Append(const std::string& key, int32_t value);
  void Append(const std::string& key, bool value);
  void Append(const std::string& key, const std::string& value);
  void Clear();

 protected:
  void Append(const std::string& key, utils::any value);

  template <typename T>
  inline std::string AppendTypeSuffix(const std::string& key) {
    return key;
  }

 private:
  //! hide the type of vector<pair<string, any>>
  utils::any data_;
};

}  // namespace common
}  // namespace cinn
