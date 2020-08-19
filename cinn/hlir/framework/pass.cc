#include "cinn/hlir/framework/pass.h"

namespace cinn {
namespace hlir {
namespace framework {

void ApplyPasses(Graph* g, const std::vector<std::string>& passes) {
  std::vector<const PassFunctionRegister*> fpass;
  for (auto& name : passes) {
    auto* reg = Registry<PassFunctionRegister>::Global()->Find(name);
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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
