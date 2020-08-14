#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cinn/hlir/framework/schedule.h"
#include "cinn/ir/packed_func.h"

using CINNCompute  = cinn::ir::PackedFunc;
using CINNSchedule = cinn::ir::PackedFunc;

namespace cinn {
namespace hlir {
namespace framework {

//! Operator implementation that includes compute and schedule function.
class OpImpl : public common::Object {
 public:
  //! Compute function
  CINNCompute fcompute;
  //! Schedule function
  CINNSchedule fschedule;
  //! Name of the implementation
  std::string name;
  //! Priority level
  int plevel;
  /**
   * \brief Invoke the operator compute function.
   * @param attrs The attribute of the primitive
   * @param inputs The input tensors.
   * @param out_type The output type information.
   * @return The output compute description of the operator.
   */
  ir::Tensor Compute(const std::vector<ir::Tensor>& inputs, const Type& out_type) {
    // TODO(haozech) : add support for packedfunc to return Tensor
    // Expected : return this->fcompute(inputs, out_type);
    ir::Tensor temp;
    return temp;
  }
  /**
   * \brief Build the computation schedule.
   * @param attrs The attribute of the node.
   * @param outs The output tensors.
   * @param target The build target.
   * @return The computation schedule.
   */
  common::Shared<Schedule> GetSchedule(const std::vector<ir::Tensor>& outs,
                                       const std::vector<ir::Tensor>& temp_tensors,
                                       const Target& target) {
    // TODO(haozech) : add support for packedfunc to return Schedule
    // Expected : return this->fschedule(outs, target);
    return nullptr;
  }

  const char* type_info() const override { return _type_key; }

 private:
  static constexpr char* _type_key = "OpImplementation";
};

//! Specialized implementations for operators under certain conditions.
class OpSpec : public common::Object {
 public:
  //! List of implementations.
  std::vector<std::shared_ptr<OpImpl>> implementations;

  /** \brief Condition to enable the specialization.
   *    Could be undefined to represent generic case.
   *  TODO(haozech) : build a specified class SpecializedCondition to represent the condition.
   *  Expected : SpecializedCondition condition;
   */
  std::string condition;

  const char* type_info() const override { return _type_key; }

  void AddImpl(CINNCompute fcompute, CINNSchedule fschedule, std::string name, int plevel) {
    auto n       = std::make_shared<OpImpl>();
    n->fcompute  = fcompute;
    n->fschedule = fschedule;
    n->name      = std::move(name);
    n->plevel    = plevel;
    this->implementations.push_back(n);
  }

 private:
  static constexpr char* _type_key = "OpSpecialization";
};

//! Operator strategy class.
class OpStrategy : public common::Object {
 public:
  const char* type_info() const override { return "CINNOpStrategy"; }
  //! List of operator specializations.
  std::vector<std::shared_ptr<OpSpec>> specializations;

  /**
   * \brief Add an implementation.
   * @param fcompute Compute function
   * @param fschedule Schedule function
   * @param name Name of the implementation
   * @param plevel Priority level of the implementation
   */
  void AddImpl(CINNCompute fcompute, CINNSchedule fschedule, std::string name, int plevel) {
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
};

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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
