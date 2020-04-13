#include "cinn/poly/dim.h"

#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace poly {

std::string Dim::range_repr() const {
  return utils::StringFormat(
      "%s <= %s <= %s", utils::GetStreamCnt(lower_bound).c_str(), id.c_str(), utils::GetStreamCnt(upper_bound).c_str());
}

Dim::Dim(std::string id, ir::Expr lower_bound, ir::Expr upper_bound)
    : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {
  optim::Simplify(&this->lower_bound);
  optim::Simplify(&this->upper_bound);
}

}  // namespace poly
}  // namespace cinn
