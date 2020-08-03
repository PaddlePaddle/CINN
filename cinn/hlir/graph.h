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

/*!
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph : public cinn::common::Graph {
 public:
  /*! \brief outputs of the computation graph. */
  std::vector<NodeData*> outputs;

  /*! \brief attributes of a graph */
  std::unordered_map<std::string, std::shared_ptr<any>> attrs;

  /*!
   * \brief Get the immutable attribute from attrs.
   * \param attr_name the name of the attribute
   * \return the reference to corresponding attribute
   * \tparam T the type of the attribute.
   */
  template <typename T>
  inline const T& GetAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    CHECK_NE(it, attrs.end()) << "Cannot find attribute " << attr_name << " in the graph";
    return get<T>(*it->second);
  }

  /*!
   * \brief Check whether has a specific attribute.
   * \param attr_name the name of the attribute
   * \return a boolean result
   */
  inline bool HasAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    return it != attrs.end();
  }
};

}  // namespace hlir
}  // namespace cinn
