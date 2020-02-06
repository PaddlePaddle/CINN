#pragma once
#include <isl/cpp.h>
#include "cinn/poly/element.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/schedule.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace poly {

class AstGen {
 public:
  AstGen(const isl::set& context) : context_(context) {}

  /**
   * Set forloop iterator names.
   * @param names
   * @return AstGen itself.
   */
  AstGen& SetIteratorNames(const std::vector<std::string>& names);

  isl::ast_node operator()(const std::vector<Element>& elements, const Scheduler& scheduler);

 private:
  isl::set context_;
  std::vector<std::string> iterator_names_;
};

}  // namespace poly
}  // namespace cinn
