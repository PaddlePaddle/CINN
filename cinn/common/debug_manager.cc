#include "cinn/common/debug_manager.h"

namespace cinn {
namespace common {

inline std::vector<std::pair<std::string, std::any>> &GetVec(std::any &data) {  // NOLINT
  return std::any_cast<std::vector<std::pair<std::string, std::any>> &>(data);
}

//! AppendTypeSuffix for multiple types.
// @{
template <>
inline std::string DebugManager::AppendTypeSuffix<int32_t>(const std::string &key) {
  return key + "_i32";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<int64_t>(const std::string &key) {
  return key + "_i64";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<float>(const std::string &key) {
  return key + "_f32";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<double>(const std::string &key) {
  return key + "_f64";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<bool>(const std::string &key) {
  return key + "_b";
}
template <>
inline std::string DebugManager::AppendTypeSuffix<std::string>(const std::string &key) {
  return key + "_s";
}
// @}

void DebugManager::Append(const std::string &key, std::any value) {
  GetVec(data_).push_back(std::make_pair(key, value));
}
void DebugManager::Append(const std::string &key, int32_t value) {
  GetVec(data_).push_back(std::make_pair(AppendTypeSuffix<int32_t>(key), value));
}
void DebugManager::Append(const std::string &key, bool value) {
  GetVec(data_).push_back(std::make_pair(AppendTypeSuffix<bool>(key), value));
}
void DebugManager::Append(const std::string &key, const std::string &value) {
  GetVec(data_).push_back(std::make_pair(AppendTypeSuffix<std::string>(key), value));
}

void DebugManager::Clear() { GetVec(data_).clear(); }

}  // namespace common
}  // namespace cinn
