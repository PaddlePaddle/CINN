#include "cinn/ir/layout.h"

namespace cinn {
namespace ir {

void Layout::Verify() {
  {
    CHECK(!name_.empty());
    CHECK(!axes_.empty());
    axis_names_ = "";
    for (auto& axis : axes_) {
      CHECK_EQ(axis->name.size(), 1U);
      auto axis_name = axis->name[0];
      CHECK((axis_name >= 'A' && axis_name <= 'Z') || (axis_name >= 'a' && axis_name <= 'z'));
      CHECK(axis_names_.find(axis_name) == axis_names_.npos) << axis_name << " has already exsit.";
      axis_names_ += axis_name;
    }
    int offset = 'A' - 'a';
    for (auto& axis : axes_) {
      CHECK_EQ(axis->name.size(), 1U);
      auto axis_name = axis->name[0];
      if (axis_name >= 'a' && axis_name <= 'z') {
        CHECK(axis_names_.find(axis_name + offset) != axis_names_.npos)
            << "sub-axis " << axis_name << " finds no primal axis";
      }
    }
  }
}
Layout::Layout(const std::string& name) {
  CHECK(!name.empty());
  int factor = 0;
  std::vector<Var> axes;
  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      CHECK_EQ(factor, 0) << "Invalid factor " << factor << " before primal axis " << c;
      axes.push_back(ir::Var(std::string(1, c)));
    } else if (c >= '0' && c <= '9') {
      factor = 10 * factor + c - '0';
    } else if (c >= 'a' && c <= 'z') {
      CHECK_GT(factor, 0) << "Invalid factor " << factor << " for sub-axis " << c;
      axes.push_back(ir::Var(factor, std::string(1, c)));
      factor = 0;
    } else {
      LOG(FATAL) << "Invalid layout: " << name;
    }
  }
  name_ = name;
  axes_ = axes;
  Verify();
}

}  // namespace ir
}  // namespace cinn
