#pragma once
#include <string>
#include <utility>
#include <vector>
#include "cinn/hlir/graph.h"
#include "cinn/utils/registry.h"

using cinn::hlir::Graph;
namespace cinn {
namespace hlir {

class PassFunctionRegister;
typedef std::function<void(Graph* g)> PassFunction;

/**
 * \brief Given an attribute of graph, find the pass that generates this attribute.
 * @param attr_name Name of the graph attribute.
 * @return The pass that generates it.
 */
const PassFunctionRegister* FindPassDep(const std::string& attr_name);

class PassFunctionRegister : public cinn::FunctionRegEntryBase<PassFunctionRegister, PassFunction> {
 public:
  bool change_structure{false};
  //! dependencies on operator attributes
  std::vector<std::string> op_attr_dependency{};
  //! dependencies on attributes in the graph
  std::vector<std::string> graph_attr_dependency{};
  //! generated targets of graph attributes
  std::vector<std::string> graph_attr_targets{};

  /**
   * \brief Imply whether this pass will change the Graph's structure.
   * @param in A bool variable implying whether this pass will change the Graph's structure.
   * @return Reference to self.
   */
  PassFunctionRegister& set_change_structure(bool in) {
    change_structure = in;
    return *this;
  }

  /**
   * \brief Declare that this pass will generate the given graph attribute name
   *        once it is applied on the graph.
   * @param attr_name Name of the graph attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& provide_graph_attr(const std::string& attr_name) {
    graph_attr_targets.push_back(attr_name);
    return *this;
  }

  /**
   * \brief Declare this pass requires the given operator attribute to be
   *        available before being applied on the graph.
   * @param attr_name Name of the attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& depend_op_attr(const std::string& attr_name) {
    op_attr_dependency.push_back(attr_name);
    return *this;
  }

  /**
   * \brief Declare this pass requires the given graph attribute to be
   *        available before being applied on the graph.
   * @param attr_name Name of the attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& depend_graph_attr(const std::string& attr_name) {
    graph_attr_dependency.push_back(attr_name);
    return *this;
  }
};

const PassFunctionRegister* FindPassDep(const std::string& attr_name) {
  for (auto* r : cinn::Registry<PassFunctionRegister>::List()) {
    for (auto& s : r->graph_attr_targets) {
      if (s == attr_name) return r;
    }
  }
  return nullptr;
}

/**
 * \brief Apply a sequence of passes on a graph.
 * @param g The input graph to apply passes on.
 * @param passes The sequence of pass.
 * @return The graph after being modified by the passes.
 */
void ApplyPasses(Graph* g, const std::vector<std::string>& passes) {
  std::vector<const PassFunctionRegister*> fpass;
  for (auto& name : passes) {
    auto* reg = cinn::Registry<PassFunctionRegister>::Find(name);
    CHECK(reg) << "Cannot find pass " << name << " in the registry";
    fpass.push_back(reg);
  }
  for (auto* r : fpass) {
    for (auto& dep : r->graph_attr_dependency) {
      CHECK_NE(g->attrs.count(dep), 0) << "To apply pass [" << r->name << "], Graph's attribute [" << dep
                                       << "] is required, but it is not available.";
      if (g->attrs.count(dep) == 0) {
        auto* pass_dep = FindPassDep(dep);
        CHECK(!pass_dep) << "And the attribute is provided by pass [" << pass_dep->name << "].";
      }
    }
    r->body(g);
  }
}

// Apply a single pass on a graph.
inline void ApplyPass(Graph* g, const std::string& pass) { return ApplyPasses(g, {pass}); }

#define CINN_REGISTER_PASS(name) CINN_REGISTRY_REGISTER(::cinn::hlir::PassFunctionRegister, PassFunctionRegister, name)
}  // namespace hlir
}  // namespace cinn
CINN_REGISTRY_ENABLE(cinn::hlir::PassFunctionRegister);
