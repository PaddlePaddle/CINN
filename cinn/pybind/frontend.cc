#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cinn/common/common.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/utils/string.h"

namespace cinn::pybind {
using common::Type;
using frontend::Placeholder;
namespace py = pybind11;
using namespace cinn::frontend;  // NOLINT

void BindFrontend(pybind11::module *m) {
  py::class_<Variable>(*m, "Variable")  //
      .def(py::init<const std::string &>(), py::arg("id") = "")
      .def("__str__", [](Variable &self) { return self->id; })
      .def("__repr__", [](Variable &self) { return utils::GetStreamCnt(self); })
      .def("set_type",
           [](Variable &self, const Type &type) {
             self->type = type;
             return self;
           })
      .def("set_shape", [](Variable &self, const std::vector<int> &shape) {
        self->shape = shape;
        return self;
      });

  py::class_<Placeholder>(*m, "Placeholder")  //
      .def(py::init<const common::Type &, const std::vector<int> &, std::string_view>(),
           py::arg("type"),
           py::arg("shape"),
           py::arg("id") = "")
      .def("shape", &Placeholder::shape)
      .def("id", &Placeholder::id)
      .def("__str__", [](const Placeholder &self) { return self.id(); });

  py::class_<Instruction>(*m, "Instruction")  //
      .def("set_attr", [](Instruction &self, const std::string &key, int x) { self.SetAttr(key, x); })
      .def("set_attr", [](Instruction &self, const std::string &key, float x) { self.SetAttr(key, x); })
      .def("set_attr", [](Instruction &self, const std::string &key, const std::string &x) { self.SetAttr(key, x); })
      .def("set_attr",
           [](Instruction &self, const std::string &key, const std::vector<int> &x) { self.SetAttr(key, x); })
      .def("set_attr",
           [](Instruction &self, const std::string &key, const std::vector<float> &x) { self.SetAttr(key, x); })
      .def("set_attr",
           [](Instruction &self, const std::string &key, const std::vector<std::string> &x) { self.SetAttr(key, x); })
      .def("get_attr_int32", &Instruction::GetAttrs<int>)
      .def("get_attr_fp32", &Instruction::GetAttrs<float>)
      .def("get_attr_str", &Instruction::GetAttrs<std::string>)
      .def("get_attr_int32s", &Instruction::GetAttrs<std::vector<int>>)
      .def("get_attr_fp32s", &Instruction::GetAttrs<std::vector<float>>)
      .def("get_attr_strs", &Instruction::GetAttrs<std::vector<std::string>>)
      .def("__str__", [](Instruction &self) { return utils::GetStreamCnt(self); });

  py::class_<Program>(*m, "Program")
      .def(py::init<>())
      .def("size", &Program::size)
      .def("__getitem__", [](Program &self, int idx) { return self[idx]; })
      .def("add", &Program::add)
      .def("relu", &Program::relu)
      .def("conv2d", &Program::conv2d)
      .def("batchnorm", &Program::batchnorm)
      .def("print_func", [](Program &self, const common::Target &target) {
        std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self));
        hlir::framework::ApplyPass(g.get(), "InferShape");
        std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
        hlir::framework::GraphCompiler gc(target, scope, g);
        gc.PrintFunc();
      });
}  // namespace frontend

}  // namespace cinn::pybind
