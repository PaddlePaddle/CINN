#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "cinn/cinn.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {
/**
 * \brief Global schedule container
 *  For operations and all the operations they depend on.
 *  The schedule per Operation is named as stage.
 */
class Schedule : public common::Object {
 public:
  const char* type_info() const override { return __type_info__; }

  /**
   * \brief Get the stage corresponds to the op
   * @param op The operation.
   */
  ir::Tensor operator[](const ir::Operation& op) {
    auto it = stage_map.find(op.name);
    CHECK(it != stage_map.end()) << "Cannot find Stage for operator " << op.name << " in the schedule";
    return it->second;
  }

  //! The output operations in original data flow graph
  std::vector<ir::Operation> outputs;
  /**
   * \brief list of all stages for ops.
   * The stages are sorted in dependency order.
   */
  std::vector<poly::Stage> stages;

  //! map of original operation to the stages
  std::unordered_map<std::string, ir::Tensor> stage_map;

 private:
  static constexpr char* __type_info__ = "CINNSchedule";
};
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
