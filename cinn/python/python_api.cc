#include "cinn/python/python_api.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cinn/python/hlir_api_wrapper.h"

namespace cinn {
namespace python {

void BindHlirApi(pybind11::module *m) {
  pybind11::class_<py_instruction> _py_instruction(*m, "Instruction");

  pybind11::class_<py_shape>(*m, "Shape")          //
      .def(pybind11::init<>())                     //
      .def("add_int_dim", &py_shape::add_int_dim)  //
      .def("add_var_dim", &py_shape::add_int_dim);

  pybind11::class_<py_context, std::shared_ptr<py_context>>(*m, "Context")  //
      .def(pybind11::init<>());

  pybind11::class_<py_computation_builder, std::shared_ptr<py_computation_builder>>(*m, "Computation")  //
      .def(pybind11::init<py_context &, const std::string &>())                                         //
      .def("add_parameter", &py_computation_builder::add_parameter)                                     //
      .def("add_binary", &py_computation_builder::add_binary)                                           //
      .def("add_dot", &py_computation_builder::add_dot);

  pybind11::class_<py_module, std::shared_ptr<py_module>>(*m, "Module")  //
      .def(pybind11::init<const std::string &>())                        //
      .def("add_computation", &py_module::add_computation)               //
      .def("add_entry_computation", &py_module::add_entry_computation);

  pybind11::class_<py_args>(*m, "Args")         //
      .def(pybind11::init<>())                  //
      .def("add_buffer", &py_args::add_buffer)  //
      .def("add_int32", &py_args::add_int32)    //
      .def("size", &py_args::size);

  pybind11::class_<py_buffer, std::shared_ptr<py_buffer>>(*m, "Buffer")    //
      .def(pybind11::init(&py_buffer::from_numpy), pybind11::arg("data"))  //
      .def("to_numpy", &py_buffer::to_numpy);

  pybind11::class_<py_compiler, std::shared_ptr<py_compiler>>(*m, "Compiler")  //
      .def(pybind11::init<>())                                                 //
      .def("compile", &py_compiler::compile)                                   //
      .def("eval", &py_compiler::eval)                                         //
      .def("eval_main", &py_compiler::eval_main);
}

}  // namespace python
}  // namespace cinn
