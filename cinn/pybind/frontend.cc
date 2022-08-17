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
#include "cinn/frontend/computation.h"
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/interpreter.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/optimize.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
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
      .def("type", [](Variable &self) { return common::Type2Str(self->type); })
      .def("set_type",
           [](Variable &self, const Type &type) {
             self->type = type;
             return self;
           })
      .def("set_type",
           [](Variable &self, const std::string &type) {
             self->type = common::Str2Type(type);
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
      .def("concat", &Program::concat)
      .def("reshape", &Program::reshape)
      .def("build_and_get_output",
           [](Program &self,
              const common::Target &target,
              const std::vector<Variable> &tensor_inputs,
              const std::vector<py::array> &input_data,
              const std::vector<Variable> &tensor_outputs) {
             std::shared_ptr<hlir::framework::Graph> g(new hlir::framework::Graph(self, target));
             hlir::framework::ApplyPass(g.get(), "InferShape");
             hlir::framework::ApplyPasses(g.get(), frontend::DefaultOpFusionPasses());
             std::shared_ptr<hlir::framework::Scope> scope = hlir::framework::BuildScope(target, g);
             hlir::framework::GraphCompiler gc(target, scope, g);
             auto program = gc.Build();
             for (size_t i = 0; i < tensor_inputs.size(); i++) {
               auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
               auto dtype     = tensor_inputs[i]->type;
               auto *data     = in_tensor->mutable_data(target, dtype);
               CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                   << "The size of tensor [" << tensor_inputs[i]->id
                   << "] is different with the input data's size! Please check.";
               if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                 CUDA_CALL(cudaMemcpy(
                     data, input_data[i].data(), in_tensor->shape().numel() * dtype.bytes(), cudaMemcpyHostToDevice));
#else
                 LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
               } else if (target.arch == Target::Arch::X86) {
                 memcpy(data, input_data[i].data(),
                        in_tensor->shape().numel() * dtype.bytes());  // All random data
               } else {
                 CINN_NOT_IMPLEMENTED
               }
             }
             program->Execute();

             std::vector<hlir::framework::Tensor> outputs;
             for (size_t i = 0; i < tensor_outputs.size(); i++) {
               outputs.push_back(scope->GetTensor(tensor_outputs[i]->id));
               outputs.back()->set_type(tensor_outputs[i]->type);
             }

             return outputs;
           })
      .def("apply_pass", &ProgramPass::Apply)

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
             VLOG(3) << info;
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
             VLOG(3) << info;
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
      .def("load_paddle_model",
           &frontend::Interpreter::LoadPaddleModel,
           py::arg("model_dir"),
           py::arg("target"),
           py::arg("params_combined"),
           py::arg("model_name") = "")
      .def("run", &frontend::Interpreter::Run)
      .def("get_tensor", &frontend::Interpreter::GetTensor)
      .def("get_program", &frontend::Interpreter::GetProgram)
      .def("get_scope", &frontend::Interpreter::GetScope);

  py::enum_<ComparisonKind>(*m, "ComparisonKind")
      .value("kUnk", ComparisonKind::kUnk)
      .value("kEq", ComparisonKind::kEq)
      .value("kNe", ComparisonKind::kNe)
      .value("kGe", ComparisonKind::kGe)
      .value("kGt", ComparisonKind::kGt)
      .value("kLe", ComparisonKind::kLe)
      .value("kLt", ComparisonKind::kLt)
      .export_values();

  py::enum_<ReduceKind>(*m, "ReduceKind")
      .value("kUnk", ReduceKind::kUnk)
      .value("kSum", ReduceKind::kSum)
      .value("kProd", ReduceKind::kProd)
      .value("kMax", ReduceKind::kMax)
      .value("kMin", ReduceKind::kMin)
      .value("kAll", ReduceKind::kAll)
      .value("kAny", ReduceKind::kAny)
      .export_values();

  py::class_<BaseBuilder>(*m, "BaseBuilder")
// clang-format off
#define PY_REGISTER_FILLCONSTANT_OP(TYPE__)                                   \
     .def("fill_constant",                                                    \
          static_cast<Variable (BaseBuilder::*)(                              \
               const std::vector<int> &, TYPE__, const std::string &, bool)>( \
               &BaseBuilder::template FillConstant<TYPE__>),                  \
          py::arg("shape"),                                                   \
          py::arg("value"),                                                   \
          py::arg("name"),                                                    \
          py::arg("force_cpu") = false)
          PY_REGISTER_FILLCONSTANT_OP(bool)
          PY_REGISTER_FILLCONSTANT_OP(float)
          PY_REGISTER_FILLCONSTANT_OP(int)
#undef PY_REGISTER_FILLCONSTANT_OP
          // clang-format on
          .def(py::init<const std::string &>(), py::arg("name") = "")
          .def("create_input",
               static_cast<Placeholder (BaseBuilder::*)(
                   const common::Type &, const std::vector<int> &, const std::string &)>(&BaseBuilder::CreateInput),
               py::arg("type"),
               py::arg("shape"),
               py::arg("id_hint") = "")
          .def("create_input", static_cast<Placeholder (BaseBuilder::*)(const Variable &)>(&BaseBuilder::CreateInput))
          .def("build", &BaseBuilder::Build, py::arg("in_reverse") = false)
          .def("name", &BaseBuilder::name)
          .def("append_instruction", &BaseBuilder::AppendInstruction)
          .def("concat", &BaseBuilder::Concat, py::arg("inputs"), py::arg("axis") = 0)
          .def("reduce",
               &BaseBuilder::Reduce,
               py::arg("a"),
               py::arg("kind")     = ReduceKind::kSum,
               py::arg("dim")      = std::vector<int>{},
               py::arg("keep_dim") = false)
          .def("fill_constant",
               static_cast<Variable (BaseBuilder::*)(
                   const std::vector<int> &, float, const std::string &, const std::string &, bool)>(
                   &BaseBuilder::FillConstant),
               py::arg("shape"),
               py::arg("value"),
               py::arg("name"),
               py::arg("dtype"),
               py::arg("force_cpu") = false)
          .def("broadcast_to",
               static_cast<Variable (BaseBuilder::*)(const Variable &, const std::vector<int> &)>(
                   &BaseBuilder::BroadcastTo),
               py::arg("a"),
               py::arg("out_shape"))
          .def("broadcast_to",
               static_cast<Variable (BaseBuilder::*)(
                   const Variable &, const std::vector<int> &, const std::vector<int> &)>(&BaseBuilder::BroadcastTo),
               py::arg("a"),
               py::arg("out_shape"),
               py::arg("broadcast_axes"))
          .def("reshape", &BaseBuilder::Reshape, py::arg("a"), py::arg("shape"))
          .def("transpose", &BaseBuilder::Transpose, py::arg("a"), py::arg("axis"))
          .def("slice",
               &BaseBuilder::Slice,
               py::arg("a"),
               py::arg("axes"),
               py::arg("starts"),
               py::arg("ends"),
               py::arg("infer_flags") = std::vector<int>{},
               py::arg("strides")     = std::vector<int>{})
          .def("reverse", &BaseBuilder::Reverse, py::arg("x"), py::arg("axis"))
          .def("select", &BaseBuilder::Select, py::arg("condition"), py::arg("true_value"), py::arg("false_value"))
          .def("split", &BaseBuilder::Split, py::arg("a"), py::arg("num_or_sections"), py::arg("axis") = 0)
          .def("index_select", &BaseBuilder::IndexSelect, py::arg("x"), py::arg("index"), py::arg("axis") = 0)
          .def("slice_assign",
               &BaseBuilder::SliceAssign,
               py::arg("input"),
               py::arg("assign"),
               py::arg("axes"),
               py::arg("starts"),
               py::arg("ends"),
               py::arg("strides") = std::vector<int>{})
          .def("scatter_assign",
               &BaseBuilder::ScatterAssign,
               py::arg("x"),
               py::arg("updates"),
               py::arg("index"),
               py::arg("axis") = 0)
          .def("scatter_add",
               &BaseBuilder::ScatterAdd,
               py::arg("x"),
               py::arg("updates"),
               py::arg("index"),
               py::arg("axis") = 0);
  ;

  py::class_<NetBuilder, BaseBuilder>(*m, "NetBuilder")
      .def(py::init<const std::string &>(), py::arg("name") = "")
  // clang-format off
#define PY_REGISTER_UNARY_FUNC(func_name__) \
  .def(SnakeName(#func_name__), &NetBuilder::func_name__, py::arg("a"))
      NETBUILDER_UNARY_OP_FOREACH(PY_REGISTER_UNARY_FUNC)
#undef PY_REGISTER_UNARY_FUNC
#define PY_REGISTER_BINARY_FUNC(func_name__) \
  .def(SnakeName(#func_name__), &NetBuilder::func_name__, py::arg("a"), py::arg("b"))
      NETBUILDER_BINARY_OP_FOREACH(PY_REGISTER_BINARY_FUNC)
#undef PY_REGISTER_BINARY_FUNC
#define PY_REGISTER_ELEMENTWISE_FUNC(func_name__) \
  .def(SnakeName(#func_name__), &NetBuilder::func_name__, py::arg("a"), py::arg("b"), py::arg("axis") = -1)
      NETBUILDER_ELEMENTWISE_OP_FOREACH(PY_REGISTER_ELEMENTWISE_FUNC)
#undef PY_REGISTER_ELEMENTWISE_FUNC
      // clang-format on
      .def("add", &NetBuilder::Add, py::arg("a"), py::arg("b"))
      .def("mul",
           &NetBuilder::Mul,
           py::arg("a"),
           py::arg("b"),
           py::arg("x_num_col_dims") = 1,
           py::arg("y_num_col_dims") = 1)
      .def("elementwise_add_grad",
           &NetBuilder::ElementwiseAddGrad,
           py::arg("dout"),
           py::arg("x"),
           py::arg("y"),
           py::arg("axis") = -1)
      .def("relu6", &NetBuilder::Relu6, py::arg("a"), py::arg("threshold") = 6.0f)
      .def("reduce_sum",
           &NetBuilder::ReduceSum,
           py::arg("x"),
           py::arg("axis")    = std::vector<int>{},
           py::arg("keepdim") = false)
      .def("all", &NetBuilder::All, py::arg("x"), py::arg("axis") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("any", &NetBuilder::Any, py::arg("x"), py::arg("axis") = std::vector<int>{}, py::arg("keepdim") = false)
      .def("conv2d",
           &NetBuilder::Conv2d,
           py::arg("a"),
           py::arg("b"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("depthwise_conv2d",
           &NetBuilder::DepthwiseConv2d,
           py::arg("a"),
           py::arg("b"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("pool2d",
           &NetBuilder::Pool2d,
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
           &NetBuilder::BatchNorm,
           py::arg("a"),
           py::arg("scale"),
           py::arg("bias"),
           py::arg("mean"),
           py::arg("variance"),
           py::arg("epsilon")     = 1e-5f,
           py::arg("momentum")    = 0.9f,
           py::arg("data_layout") = "NCHW",
           py::arg("is_test")     = true)
      .def("batch_norm_grad",
           &NetBuilder::BatchNormGrad,
           py::arg("dy"),
           py::arg("x"),
           py::arg("scale"),
           py::arg("save_mean"),
           py::arg("save_variance"),
           py::arg("epsilon")     = 1e-5,
           py::arg("data_layout") = "NCHW")
      .def("scale",
           &NetBuilder::Scale,
           py::arg("a"),
           py::arg("scale")            = 1.0f,
           py::arg("bias")             = 0.0f,
           py::arg("bias_after_scale") = true)
      .def("softmax", &NetBuilder::Softmax, py::arg("a"), py::arg("axis") = -1, py::arg("data_format") = "AnyLayout")
      .def("dropout_infer",
           &NetBuilder::DropoutInfer,
           py::arg("a"),
           py::arg("dropout_prob")           = 0.5f,
           py::arg("dropout_implementation") = "downgrade_in_infer")
      .def("conv2d_grad",
           &NetBuilder::Conv2dGrad,
           py::arg("dy"),
           py::arg("x"),
           py::arg("w"),
           py::arg("strides")           = std::vector<int>{1, 1},
           py::arg("paddings")          = std::vector<int>{0, 0},
           py::arg("dilations")         = std::vector<int>{1, 1},
           py::arg("groups")            = 1,
           py::arg("data_format")       = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("sum", &NetBuilder::Sum, py::arg("inputs"))
      .def("matmul",
           &NetBuilder::Matmul,
           py::arg("x"),
           py::arg("y"),
           py::arg("transpose_x") = false,
           py::arg("transpose_y") = false,
           py::arg("alpha")       = 1.0f);

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
           py::arg("output_shape")      = std::vector<int>{})
      .def("compare", &CinnBuilder::Compare, py::arg("lhs"), py::arg("rhs"), py::arg("kind") = ComparisonKind::kEq)
      .def("__str__", [](CinnBuilder &self) { return self.name(); });

  auto computation = py::class_<CinnComputation, std::shared_ptr<CinnComputation>>(*m, "Computation");
  py::class_<CinnComputation::CompileOptions>(computation, "CompileOptions")
      .def_readwrite("use_decomposer", &CinnComputation::CompileOptions::use_decomposer)
      .def_readwrite("do_prerun", &CinnComputation::CompileOptions::do_prerun)
      .def_readwrite("use_default_passes", &CinnComputation::CompileOptions::use_default_passes)
      .def_readwrite("passes", &CinnComputation::CompileOptions::passes);

  computation
      .def("default_compile_options", &CinnComputation::DefaultCompileOptions)
      // currently stream param is not exported to python, the default stream is used always
      .def_static(
          "build_and_compile",
          [](const common::Target &target, BaseBuilder &builder, const CinnComputation::CompileOptions &options) {
            return CinnComputation::BuildAndCompile(target, builder, options);
          },
          py::arg("target"),
          py::arg("builder"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def_static(
          "compile",
          [](const common::Target &target, Program &program, const CinnComputation::CompileOptions &options) {
            return CinnComputation::Compile(target, program, options);
          },
          py::arg("target"),
          py::arg("program"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def_static(
          "compile_paddle_model",
          [](const common::Target &target,
             const std::string &model_path,
             const std::vector<std::string> &input_names,
             const std::vector<hlir::framework::shape_t> &input_shapes,
             bool params_combined,
             const CinnComputation::CompileOptions &options) {
            return CinnComputation::CompilePaddleModel(
                target, model_path, input_names, input_shapes, params_combined, options);
          },
          py::arg("target"),
          py::arg("model_path"),
          py::arg("input_names"),
          py::arg("input_shapes"),
          py::arg("params_combined"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def("get_all_tensor_names", &CinnComputation::GetAllTensorNames)
      .def("get_tensor", &CinnComputation::GetTensor)
      .def("execute", [](CinnComputation &self) { self.Execute(); });

}  // namespace frontend

}  // namespace cinn::pybind
