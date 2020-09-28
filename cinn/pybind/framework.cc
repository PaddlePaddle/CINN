#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cinn/common/cinn_value.h"
#include "cinn/frontend/executor.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/framework/scope.h"
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
              const std::vector<std::vector<int>> &output_shapes,
              const common::Target &target) {
             const Operator *op_ptr = Operator::Get(key);
             auto impl = OpStrategy::SelectImpl(self[op_ptr](attrs, inputs, out_types, output_shapes, target));
             std::vector<common::CINNValue> temp_inputs;
             std::vector<ir::Tensor> res;
             for (auto &tensor : inputs) {
               res.push_back(tensor);
               temp_inputs.push_back(common::CINNValue(tensor));
             }
             common::CINNValuePack C = impl->fcompute(common::CINNValuePack{temp_inputs});
             poly::StageMap stages   = C.back();
             // make sure all the tensors in the stages before schedule launch.
             for (int i = 0; i < C->size() - 1; i++) {
               ir::Expr temp = C[i];
               stages->InsertLazily(temp.as_tensor_ref());
             }
             C = impl->fschedule(C);
             for (int i = 0; i < C->size() - 1; i++) {
               ir::Expr temp = C[i];
               res.push_back(temp.as_tensor_ref());
             }
             auto func = Lower(key, stages, res);
             return func;
           });

  py::class_<NodeAttr>(*m, "NodeAttr")
      .def(py::init<>())
      .def_readwrite("attr_store", &NodeAttr::attr_store)
      .def("set_attr",
           [](NodeAttr &self, const std::string &key, NodeAttr::attr_t value) { self.attr_store[key] = value; })
      .def("get_attr",
           [](NodeAttr &self, const std::string &key) {
             CHECK_EQ(self.attr_store.count(key), 1) << "Didn't find value with key [" << key << "].";
             return self.attr_store[key];
           })
      .def("__str__", [](NodeAttr &self) { return utils::GetStreamCnt(self); });

  py::class_<Scope>(*m, "Scope")
      .def(py::init<>())  //
      .def("get_tensor",
           [](Scope &self, const std::string &name) {
             py::dtype dt = py::dtype::of<float>();
             auto t       = self.GetTensor(name);
             py::array::ShapeContainer shape(t->shape().data().begin(), t->shape().data().end());
             py::array array(std::move(dt), std::move(shape));
             auto *mutable_data = array.mutable_data();
             std::memcpy(mutable_data, t->data<float>(), t->shape().numel() * sizeof(float));
             return array;
           })
      .def("var_names", &Scope::var_names);

  py::class_<common::Shared<hlir::framework::_Tensor_>>(*m, "SharedTensor");
  py::class_<Tensor, common::Shared<hlir::framework::_Tensor_>>(*m, "Tensor")
      .def(py::init<>())
      .def("shape", [](hlir::framework::Tensor &self) { return self->shape().data(); })
      .def("set_type", [](hlir::framework::Tensor &self, Type type) { self->set_type(type); })
      .def("numpy",
           [](hlir::framework::Tensor &self) {
             py::dtype dt;
             // set float by default
             dt = py::dtype::of<float>();
             py::array::ShapeContainer shape(self->shape().data().begin(), self->shape().data().end());
             py::array array(std::move(dt), std::move(shape));
             void *array_data = array.mutable_data();
             std::memcpy(array_data, self->data<float>(), self->shape().numel() * sizeof(float));
             return array;
           })
      .def("from_numpy", [](hlir::framework::Tensor &self, py::array array) {
        CHECK(array.dtype().is(py::dtype::of<float>())) << "currently only support float32 data type as input";
        hlir::framework::shape_t shape;
        std::copy_n(array.shape(), array.ndim(), std::back_inserter(shape));
        self->Resize(Shape(shape));
        // TODO(Superjomn) Support other target.
        auto *data = self->mutable_data<float>(common::DefaultHostTarget());
        for (int i = 0; i < self->shape().numel(); i++) {
          data[i] = reinterpret_cast<const float *>(array.data())[i];
        }
      });
}
}  // namespace cinn::pybind
