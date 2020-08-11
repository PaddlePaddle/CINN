#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cinn/hlir/schedule.h"
#include "cinn/ir/packed_func.h"

using CINNCompute  = cinn::ir::PackedFunc;
using CINNSchedule = cinn::ir::PackedFunc;

namespace cinn {
namespace hlir {

//! Operator implementation that includes compute and schedule function.
class OpImplementation : public common::Object {
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
    // To do : add support for packedfunc to return Tensor
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
  hlir::Schedule* Schedule(const std::vector<ir::Tensor>& outs, const Target& target) {
    // To do : add support for packedfunc to return Schedule
    // Expected : return this->fschedule(outs, target);
    hlir::Schedule* temp;
    return temp;
  }

  const char* type_info() const { return _type_key; }

 private:
  static constexpr const char* _type_key = "OpImplementation";
};

//! Specialized implementations for operators under certain conditions.
class OpSpecialization : public common::Object {
 public:
  //! List of implementations.
  std::vector<OpImplementation*> implementations;

  /** \brief Condition to enable the specialization.
   *    Could be undefined to represent generic case.
   *  To do : build a specified class SpecializedCondition to represent the condition.
   *  Expected : SpecializedCondition condition;
   */
  std::string condition;

  const char* type_info() const { return _type_key; }

  void AddImplementation(CINNCompute fcompute, CINNSchedule fschedule, std::string name, int plevel) {
    auto n       = make_shared<OpImplementation>();
    n->fcompute  = fcompute;
    n->fschedule = fschedule;
    n->name      = std::move(name);
    n->plevel    = plevel;
    this->implementations.push_back(n);
  }

 private:
  static constexpr const char* _type_key = "OpSpecialization";
};

//! Operator strategy class.
class OpStrategy : public common::Object {
 public:
  const char* type_info() const override { return "CINNOpStrategy"; }
  //! List of operator specializations.
  std::vector<OpSpecialization*> specializations;

  /**
   * \brief Add an implementation.
   * @param fcompute Compute function
   * @param fschedule Schedule function
   * @param name Name of the implementation
   * @param plevel Priority level of the implementation
   */
  void AddImplementation(CINNCompute fcompute, CINNSchedule fschedule, std::string name, int plevel) {
    //! To do : here curr_cond should get the condition from outside.
    //! Expected : auto curr_cond = SpecializedCondition::Current();
    std::string curr_cond = "current_condition";
    OpSpecialization* op_spec;
    for (OpSpecialization* op_spec : specializations) {
      if (op_spec->condition == curr_cond) {
        op_spec->AddImplementation(fcompute, fschedule, std::move(name), plevel);
        return;
      }
    }
    OpSpecialization* n = make_shared<OpSpecialization>();
    n->condition        = curr_cond;
    n->AddImplementation(fcompute, fschedule, std::move(name), plevel);
    this->specializations.push_back(n);
  }
};

}  // namespace hlir
}  // namespace cinn
