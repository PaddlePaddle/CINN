#pragma once
#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cinn {
namespace hlir {
namespace instruction {

struct Context {
  inline int new_ssa_id() { return ssa_count_++; }
  inline std::string new_ssa_id(const std::string& hint) { return hint + std::to_string(key_[hint]++); }

  inline std::string new_computation_id(const std::string& hint) {
    if (!hint.empty()) {
      CHECK(!computation_names_.count(hint)) << "duplicate computation name " << hint;
      return hint;
    }
    return "computation" + std::to_string(computation_counter_++);
  }

  inline std::string new_var_name(const std::string& hint = "_v") { return hint + std::to_string(var_count_++); }

 private:
  int ssa_count_{0};
  std::unordered_map<std::string, size_t> key_;
  std::unordered_set<std::string> computation_names_;
  int computation_counter_{0};
  size_t var_count_{0};
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
