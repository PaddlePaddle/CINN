// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "cinn/common/common.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/interpreter.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/pybind/bind.h"
#include "cinn/utils/string.h"
#include "cinn/utils/timer.h"

namespace cinn::pybind {
using common::Type;
using frontend::Placeholder;
namespace py = pybind11;
using namespace cinn::frontend;  // NOLINT

// this function is a helper function, not threadsafe,
// used in this file only for py function register
static const char *SnakeName(const char *name) {
  static char buf[256];
  char *p       = buf;
  const char *q = name;
  for (; *q; q++, p++) {
    if ((*q >= 'A') && (*q <= 'Z')) {
      if (p > buf) *p++ = '_';
      *p = *q - 'A' + 'a';
    } else {
      *p = *q;
    }
  }
  *p = 0;
  return buf;
}

void BindFrontend(pybind11::module *m) {
  py::class_<Variable>(*m, "Variable")  //
      .def(py::init<const std::string &>(), py::arg("id") = "")
      .def(py::init([](const Placeholder &p) { return new Variable(p); }))
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

  py::implicitly_convertible<Placeholder, Variable>();

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

  py::class_<BaseBuilder>(*m, "BaseBuilder")
      .def(py::init<const std::string &>(), py::arg("name") = "")
      .def("create_input",
           static_cast<Placeholder (BaseBuilder::*)(
               const common::Type &, const std::vector<int> &, const std::string &)>(&BaseBuilder::CreateInput),
           py::arg("type"),
           py::arg("shape"),
           py::arg("id_hint") = "")
      .def("create_input", static_cast<Placeholder (BaseBuilder::*)(const Variable &)>(&BaseBuilder::CreateInput))
      .def("build", &BaseBuilder::Build)
      .def("name", &BaseBuilder::name)
      .def("append_instruction", &BaseBuilder::AppendInstruction);

  py::class_<NetBuilder, BaseBuilder>(*m, "NetBuilder")
      .def(py::init<const std::string &>(), py::arg("name") = "")
      .def("add", &NetBuilder::add, py::arg("a"), py::arg("b"))
      .def("mul",
           &NetBuilder::mul,
           py::arg("a"),
           py::arg("b"),
           py::arg("x_num_col_dims") = 1,
           py::arg("y_num_col_dims") = 1)
      .def("mulbias",
           &NetBuilder::mulbias,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("x_num_col_dims") = 1,
           py::arg("y_num_col_dims") = 1)
      .def("elementwise_add", &NetBuilder::elementwise_add, py::arg("a"), py::arg("b"), py::arg("axis") = -1)
      .def("elementwise_mul", &NetBuilder::elementwise_mul, py::arg("a"), py::arg("b"), py::arg("axis") = -1)
      .def("relu", &NetBuilder::relu, py::arg("a"))
      .def("relu_grad", &NetBuilder::relu_grad, py::arg("dout"), py::arg("out"))
      .def("relu6", &NetBuilder::relu6, py::arg("a"), py::arg("threshold") = 6.0f)
      .def("conv2d",
           &NetBuilder::conv2d,
           py::arg("a"),
           py::arg("b"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("depthwise_conv2d",
           &NetBuilder::depthwise_conv2d,
           py::arg("a"),
           py::arg("b"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("pool2d",
           &NetBuilder::pool2d,
           py::arg("a"),
           py::arg("polling_type"),
           py::arg("ksize"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("ceil_mode")         = false,
           py::arg("exclusive")         = true,
           py::arg("global_pooling")    = false,
           py::arg("data_format")       = "HCHW",
           py::arg("adaptive")          = false,
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("batchnorm",
           &NetBuilder::batchnorm,
           py::arg("a"),
           py::arg("scale"),
           py::arg("bias"),
           py::arg("mean"),
           py::arg("variance"),
           py::arg("epsilon")     = 1e-5f,
           py::arg("momentum")    = 0.9f,
           py::arg("data_layout") = "NCHW")
      .def("scale",
           &NetBuilder::scale,
           py::arg("a"),
           py::arg("scale")            = 1.0f,
           py::arg("bias")             = 0.0f,
           py::arg("bias_after_scale") = true)
      .def("softmax", &NetBuilder::softmax, py::arg("a"), py::arg("axis") = -1, py::arg("data_format") = "AnyLayout")
      .def("sigmoid", &NetBuilder::sigmoid, py::arg("a"))
      .def("slice",
           &NetBuilder::slice,
           py::arg("a"),
           py::arg("axes"),
           py::arg("starts")        = std::vector<int>{},
           py::arg("ends")          = std::vector<int>{},
           py::arg("infer_flags")   = std::vector<int>(),
           py::arg("decrease_axis") = std::vector<int>())
      .def("dropout_infer",
           &NetBuilder::dropout_infer,
           py::arg("a"),
           py::arg("dropout_prob")           = 0.5f,
           py::arg("dropout_implementation") = "downgrade_in_infer")
      .def("conv2d_grad",
           &NetBuilder::conv2d_grad,
           py::arg("dy"),
           py::arg("x"),
           py::arg("w"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT");

  py::class_<CinnBuilder, BaseBuilder>(*m, "CinnBuilder")
      .def(py::init<const std::string &>(), py::arg("name") = "")
      .def("const_scalar", &CinnBuilder::ConstScalar<bool>)
      .def("const_scalar", &CinnBuilder::ConstScalar<float>)
      .def("const_scalar", &CinnBuilder::ConstScalar<int>)
  // clang-format off
#define PY_REGISTER_FUNC(func_name__) .def(SnakeName(#func_name__), &CinnBuilder::func_name__)
          UNARY_OP_FOREACH(PY_REGISTER_FUNC)
          BINARY_OP_FOREACH(PY_REGISTER_FUNC)
#undef PY_REGISTER_FUNC
      // clang-format on
      .def("concat", &CinnBuilder::Concat, py::arg("lhs"), py::arg("rhs"), py::arg("axis") = 0)
      .def("conv",
           &CinnBuilder::Conv,
           py::arg("lhs"),
           py::arg("rhs"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("conv_type")         = "forward",
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT",
           py::arg("filter_shape")      = std::vector<int>{})
      .def("compare", &CinnBuilder::Compare, py::arg("lhs"), py::arg("rhs"), py::arg("kind"))
      .def("reduce",
           &CinnBuilder::Reduce,
           py::arg("operand"),
           py::arg("kind"),
           py::arg("dim"),
           py::arg("keep_dim") = false)
      .def("broadcast_to",
           &CinnBuilder::BroadcastTo,
           py::arg("operand"),
           py::arg("out_shape"),
           py::arg("broadcast_axes"))
      .def("reshape", &CinnBuilder::Reshape, py::arg("operand"), py::arg("shape"))
      .def("slice",
           &CinnBuilder::Slice,
           py::arg("operand"),
           py::arg("axes"),
           py::arg("starts") = std::vector<int>{},
           py::arg("ends")   = std::vector<int>{})
      .def("select", &CinnBuilder::Select, py::arg("condition"), py::arg("true_value"), py::arg("false_value"))
      .def("reverse", &CinnBuilder::Reverse, py::arg("operand"), py::arg("axis"))
      .def("__str__", [](CinnBuilder &self) { return self.name(); });

}  // namespace frontend

}  // namespace cinn::pybind
