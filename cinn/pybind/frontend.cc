#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "cinn/pybind/bind.h"
#include "cinn/common/common.h"
#include "cinn/frontend/interpreter.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/utils/string.h"
#include "cinn/utils/timer.h"

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
      .def(py::init<const common::Type &, const std::vector<int> &, absl::string_view>(),
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
      .def("mul", &Program::mul)
      .def("mulbias", &Program::mulbias)
      .def("elementwise_add", &Program::elementwise_add)
      .def("relu", &Program::relu)
      .def("relu6", &Program::relu6)
      .def("sigmoid", &Program::sigmoid)
      .def("dropout_infer", &Program::dropout_infer)
      .def("scale", &Program::scale)
      .def("slice", &Program::slice)
      .def("conv2d", &Program::conv2d)
      .def("depthwise_conv2d", &Program::depthwise_conv2d)
      .def("batchnorm", &Program::batchnorm)
      .def("softmax", &Program::softmax)
      .def("pool2d", &Program::pool2d)
      .def("build_and_get_output",
           [](Program &self,
              const common::Target &target,
              const std::vector<Variable> &tensor_inputs,
              const std::vector<py::array> &input_data,
              const Variable &tensor_out) {
             std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self, target));
             hlir::framework::ApplyPass(g.get(), "InferShape");
             if (target.arch == Target::Arch::NVGPU) {
               hlir::framework::ApplyPass(g.get(), "OpFusion");
             }
             std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
             hlir::framework::GraphCompiler gc(target, scope, g);
             auto program = gc.Build();
             for (size_t i = 0; i < tensor_inputs.size(); i++) {
               auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
               auto *data     = in_tensor->mutable_data<float>(target);
               CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                   << "The size of tensor [" << tensor_inputs[i]->id
                   << "] is different with the input data's size! Please check.";
               if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                 CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                      input_data[i].data(),
                                      in_tensor->shape().numel() * sizeof(float),
                                      cudaMemcpyHostToDevice));
#else
                 LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
               } else if (target.arch == Target::Arch::X86) {
                 for (size_t j = 0; j < in_tensor->shape().numel(); j++) {
                   data[j] = reinterpret_cast<const float *>(input_data[i].data())[j];  // All random data
                 }
               } else {
                 CINN_NOT_IMPLEMENTED
               }
             }
             program->Execute();
             auto out = scope->GetTensor(tensor_out->id);
             return out;
           })
      /**
       * @brief Test the performance of a single-op program
       * @param self The program built with only one op
       * @param target The Target that controls the backends to execute on
       * @param tensor_inputs The vector that contains all input Variables. Must be on CPU
       * @param input_data The vector that contains each input Variable's data(stored as py::array)
       * @param tensor_out The output Variable.
       * @param repeat_ The number of executing time. Increase it to avoid testing noise.
       * @param info The string to be print before testing. Usually it implyies the kind of op and
       *  input variable's shape.
       *
       * @return The output tensor after executing the op.
       *
       * @note
       *  This function is for user to test single op performance on python.
       *  To learn more about how to test op's benchmark, see '/python/tests/test_op_benchmark.py'
       *
       */
      .def("test_benchmark",
           [](Program &self,
              const common::Target &target,
              const std::vector<Variable> &tensor_inputs,
              const std::vector<py::array> &input_data,
              const Variable &tensor_out,
              int repeat_,
              const std::string &info) {
             std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self, target));
             hlir::framework::ApplyPass(g.get(), "InferShape");
             std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
             hlir::framework::GraphCompiler gc(target, scope, g);
             auto program = gc.Build();
             for (size_t i = 0; i < tensor_inputs.size(); i++) {
               auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
               auto *data     = in_tensor->mutable_data<float>(target);
               CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                   << "The size of tensor [" << tensor_inputs[i]->id
                   << "] is different with the input data's size! Please check.";
               if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                 CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                      input_data[i].data(),
                                      in_tensor->shape().numel() * sizeof(float),
                                      cudaMemcpyHostToDevice));
#else
                 LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
               } else if (target.arch == Target::Arch::X86) {
                 for (size_t j = 0; j < in_tensor->shape().numel(); j++) {
                   data[j] = reinterpret_cast<const float *>(input_data[i].data())[j];  // All random data
                 }
               } else {
                 CINN_NOT_IMPLEMENTED
               }
             }
             LOG(INFO) << info;
             program->ExecuteTest(repeat_);
             auto out = scope->GetTensor(tensor_out->id);
             return out;
           })
      .def("test_benchmark_with_code",
           [](Program &self,
              const common::Target &target,
              const std::vector<Variable> &tensor_inputs,
              const std::vector<py::array> &input_data,
              const Variable &tensor_out,
              int repeat_,
              const std::string &info,
              const std::string &code) {
             std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self, target));
             hlir::framework::ApplyPass(g.get(), "InferShape");
             std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
             hlir::framework::GraphCompiler gc(target, scope, g);
             auto program = gc.Build(code);
             for (size_t i = 0; i < tensor_inputs.size(); i++) {
               auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
               auto *data     = in_tensor->mutable_data<float>(target);
               CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                   << "The size of tensor [" << tensor_inputs[i]->id
                   << "] is different with the input data's size! Please check.";
               if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                 CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                      input_data[i].data(),
                                      in_tensor->shape().numel() * sizeof(float),
                                      cudaMemcpyHostToDevice));
#else
                 LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
               } else if (target.arch == Target::Arch::X86) {
                 for (size_t j = 0; j < in_tensor->shape().numel(); j++) {
                   data[j] = reinterpret_cast<const float *>(input_data[i].data())[j];  // All random data
                 }
               } else {
                 CINN_NOT_IMPLEMENTED
               }
             }
             LOG(INFO) << info;
             program->ExecuteTest(repeat_);
             auto out = scope->GetTensor(tensor_out->id);
             return out;
           })
      .def("test_generate_code",
           [](Program &self,
              const common::Target &target,
              const std::vector<Variable> &tensor_inputs,
              const std::vector<py::array> &input_data,
              const Variable &tensor_out) {
             std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self, target));
             hlir::framework::ApplyPass(g.get(), "InferShape");
             std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
             hlir::framework::GraphCompiler gc(target, scope, g);
             return gc.GenSourceCode();
           });

  py::class_<frontend::Interpreter>(*m, "Interpreter")
      .def(py::init<const std::vector<std::string> &, const std::vector<hlir::framework::shape_t> &>(),
           py::arg("input_names"),
           py::arg("input_shapes"))  //
      .def("load_paddle_model", &frontend::Interpreter::LoadPaddleModel)
      .def("run", &frontend::Interpreter::Run)
      .def("get_tensor", &frontend::Interpreter::GetTensor)
      .def("scope", &frontend::Interpreter::scope);
}  // namespace frontend

}  // namespace cinn::pybind
