#pragma once
#include <set>
#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"

namespace cinn {
namespace ir {
class Layout {
 public:
  std::string name_;
  std::string axis_names_;
  std::vector<ir::Var> axes_;

  Layout(const std::string& name, const std::vector<ir::Var>& axes) : name_(name), axes_(axes) { Verify(); }

  explicit Layout(const std::string& name);

  inline const std::string& name() const { return name_; }
  // axis name without factor
  inline const std::string& axis_names() const { return axis_names_; }
  inline const std::vector<ir::Var>& axes() const { return axes_; }
  inline int ndims() const { return axes_.size(); }
  inline const Var operator[](int i) const { return axes_[i]; }
  inline const char axis_names(int i) const { return axis_names_[i]; }

  void Verify();
  Expr Make(const std::string& name, const std::vector<ir::Var>& axes);
};

}  // namespace ir
}  // namespace cinn
