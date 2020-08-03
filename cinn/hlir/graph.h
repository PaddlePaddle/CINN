#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "cinn/common/graph_utils.h"
#include "cinn/hlir/node.h"
using cinn::utils::any;
using cinn::utils::get;

namespace cinn {
namespace hlir {
// computation graph
class CINN_Graph : public cinn::common::Graph {
 public:
  std::vector<NodeData*> outputs;
  std::unordered_map<std::string, std::shared_ptr<any>> attrs;
  template <typename T>
  inline const T& GetAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    CHECK_NE(it, attrs.end()) << "Cannot find attribute " << attr_name << " in the graph";
    return get<T>(*it->second);
  }
  inline bool HasAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    return it != attrs.end();
  }
};

}  // namespace hlir
}  // namespace cinn
