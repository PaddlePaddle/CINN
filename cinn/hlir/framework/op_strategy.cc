#include "cinn/hlir/framework/op_strategy.h"
namespace cinn {
namespace hlir {
namespace framework {

std::shared_ptr<OpImpl> SelectImpl(std::shared_ptr<OpStrategy> strategy) {
  //! should get the host info from global environment.
  std::string curr_condition  = "default";
  std::shared_ptr<OpImpl> res = nullptr;
  for (auto spec : strategy->specializations) {
    if (spec->condition == "default") {
      for (auto i : spec->implementations) {
        if (!res || res->plevel < i->plevel) {
          res = i;
        }
      }
    }
  }
  CHECK(res) << "There is no available strategy implementation! SelectImpl failed!";
  return res;
}

void OpStrategy::AddImpl(CINNCompute fcompute, CINNSchedule fschedule, std::string name, int plevel) {
  //! TODO(haozech) : here curr_cond should get the condition from outside.
  //! Expected : auto curr_cond = SpecializedCondition::Current();
  std::string curr_condition = "default";
  for (auto op_spec : specializations) {
    if (op_spec->condition == curr_condition) {
      op_spec->AddImpl(fcompute, fschedule, std::move(name), plevel);
      return;
    }
  }
  std::shared_ptr<OpSpec> n = std::make_shared<OpSpec>();
  n->condition              = curr_condition;
  n->AddImpl(fcompute, fschedule, std::move(name), plevel);
  this->specializations.push_back(n);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
