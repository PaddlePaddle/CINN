#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cinn/common/cinn_value.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"

namespace cinn::pybind {

namespace py = pybind11;
using namespace cinn::hlir::framework;  // NOLINT

void BindFramework(pybind11::module *m) {
  py::class_<Operator>(*m, "Operator").def("get_op_attrs", [](const std::string &key) {
    return Operator::GetAttrs<StrategyFunction>(key);
  });

  py::class_<OpValueType<StrategyFunction>>(*m, "OpValueType")
      .def("apply_strategy",
           [](OpValueType<StrategyFunction> &self,
              const std::string &key,
              const NodeAttr &attrs,
              const std::vector<ir::Tensor> &inputs,
              const std::vector<Type> &out_types,
              const common::Target &target) {
             const Operator *op_ptr = Operator::Get(key);
             auto impl              = OpStrategy::SelectImpl(self[op_ptr](attrs, inputs, out_types, target));
             std::vector<common::CINNValue> temp_inputs;
             std::vector<ir::Tensor> res;
             for (auto tensor : inputs) {
               res.push_back(tensor);
               temp_inputs.push_back(common::CINNValue(tensor));
             }
             auto stages = CreateStages(inputs);
             temp_inputs.push_back(common::CINNValue(stages));
             common::CINNValuePack C = impl->fcompute(common::CINNValuePack{temp_inputs});
             C                       = impl->fschedule(C);
             for (int i = 0; i < C.get()->size() - 1; i++) {
               ir::Expr temp = C[i];
               res.push_back(temp.as_tensor_ref());
             }
             return res;
           });

  py::class_<NodeAttr>(*m, "NodeAttr")
      .def(py::init<>())
      .def_readwrite("attr_store", &NodeAttr::attr_store)
      .def("set_attr",
           [](NodeAttr &self, const std::string &key, NodeAttr::attr_t value) { self.attr_store[key] = value; })
      .def("__str__", [](NodeAttr &self) { return utils::GetStreamCnt(self); });

}  // namespace frontend
}  // namespace cinn::pybind
