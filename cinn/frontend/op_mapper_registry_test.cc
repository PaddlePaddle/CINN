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

#include "cinn/frontend/op_mapper_registry.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/frontend/decomposer/test_helper.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/paddle_model_to_program.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/utils/registry.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {

using ::cinn::common::Target;
using ::cinn::common::Type;
using ::cinn::frontend::paddle::cpp::OpDesc;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;
using ::cinn::utils::TransValidVarName;

std::default_random_engine engine(/* seed */ 0);

class OpMapperTestUtil {
 public:
  static void AssertOpMapReg_Same_ModelToProg(const OpDesc& op_desc,
                                              const std::unordered_map<std::string, void*>& input_data,
                                              const std::unordered_map<std::string, Type>& input_types,
                                              const std::unordered_map<std::string, std::vector<int>>& input_shapes) {
#ifdef CINN_WITH_CUDA
    Target target = common::DefaultNVGPUTarget();
#else
    Target target = common::DefaultHostTarget();
#endif

    std::unordered_map<std::string, std::vector<float>> op_map_res =
        GetOpMapRegResult(op_desc, input_data, input_types, input_shapes);
    std::unordered_map<std::string, std::vector<float>> model_to_prog_res =
        GetModelToProgResult(op_desc, input_data, input_types, input_shapes);

    ASSERT_EQ(op_map_res.size(), model_to_prog_res.size());
    // outputs variables have different names, so we have to search for map
    for (const auto& res : op_map_res) {
      bool has_map = false;
      for (auto iter = model_to_prog_res.begin(); iter != model_to_prog_res.end(); ++iter) {
        if (res.second == iter->second) {
          has_map = true;
          model_to_prog_res.erase(iter);
          break;
        }
      }
      ASSERT_TRUE(has_map) << res.first << " doesn't have result in model_to_prog_res.";
    }
  }

 private:
  template <typename T>
  static void SetTensorData(const void* data, const Target& target, Tensor tensor) {
    size_t num_ele = tensor->shape().numel();
    T* tensor_data = tensor->mutable_data<T>(target);
#ifdef CINN_WITH_CUDA
    cudaMemcpy(tensor_data, data, num_ele * sizeof(T), cudaMemcpyHostToDevice);
#else
    memcpy(tensor_data, data, num_ele * sizeof(T));
#endif
  }

  static void SetTensorData(const void* data, const Target& target, const Type& type, Tensor tensor) {
    if (type == cinn::common::F32()) {
      SetTensorData<float>(data, target, tensor);
    } else if (type == ::cinn::common::I32()) {
      SetTensorData<int>(data, target, tensor);
    } else {
      LOG(FATAL) << "We don't support " << type << " now";
    }
  }

  static std::unordered_set<std::string> GetOutputNames(const Program& program) {
    std::unordered_set<std::string> output_names;
    for (size_t i = 0; i < program.size(); ++i) {
      const auto& instr   = program[i];
      const auto& outputs = instr.GetOutputs();
      for (const auto& var : outputs) {
        output_names.insert(var->id);
        // Some ops such as conv2d has different outputs for different target
        // in one instruction, but PaddleModelToProgram only adds Output(0)
        // to build program. We just compare the first output now.
        // Remove the break after fixing.
        break;
      }
      const auto& inputs = instr.GetInputs();
      for (const auto& var : inputs) {
        if (output_names.count(var->id) != 0) {
          output_names.erase(var->id);
        }
      }
    }
    return output_names;
  }

  static void SetInputToScope(const std::unordered_map<std::string, void*>& input_data,
                              const std::unordered_map<std::string, Type>& input_types,
                              const std::unordered_map<std::string, std::vector<int>>& input_shapes,
                              const Target& target,
                              std::shared_ptr<Scope> scope) {
    for (const auto& input_data_pair : input_data) {
      const std::string& input_name       = input_data_pair.first;
      const Type& input_type              = input_types.at(input_name);
      const std::vector<int>& input_shape = input_shapes.at(input_name);
      scope->Var<Tensor>(input_name);
      Tensor input_tensor = scope->GetTensor(input_name);
      input_tensor->Resize(Shape(input_shape));
      SetTensorData(input_data_pair.second, target, input_type, input_tensor);
    }
  }

  static std::unordered_map<std::string, std::vector<float>> GetOutputFromScope(
      const std::unordered_set<std::string> output_names, std::shared_ptr<Scope> scope) {
    std::unordered_map<std::string, std::vector<float>> ret;
    for (const std::string& name : output_names) {
      Tensor output_tensor = scope->GetTensor(name);
      if (output_tensor->type() == ::cinn::common::F32()) {
        std::vector<float> out_data(output_tensor->shape().numel());
        CopyToVector(output_tensor, &out_data);
        ret[name] = out_data;
      } else if (output_tensor->type() == ::cinn::common::I32()) {
        std::vector<int> out_data(output_tensor->shape().numel());
        CopyToVector(output_tensor, &out_data);
        ret[name] = std::vector<float>(out_data.begin(), out_data.end());
      } else {
        LOG(FATAL) << "We don't support " << output_tensor->type() << " now";
      }
    }
    return ret;
  }

  // Assert macro returns, so we have to put it in void function
  static void CheckOpMapperNotNull(const OpMapper* op_mapper) { ASSERT_NE(op_mapper, nullptr); }

  static std::unordered_map<std::string, std::vector<float>> GetOpMapRegResult(
      const OpDesc& op_desc,
      const std::unordered_map<std::string, void*>& input_data,
      const std::unordered_map<std::string, Type>& input_types,
      const std::unordered_map<std::string, std::vector<int>>& input_shapes) {
#ifdef CINN_WITH_CUDA
    Target target = common::DefaultNVGPUTarget();
#else
    Target target = common::DefaultHostTarget();
#endif
    std::shared_ptr<Scope> ctx_scope = Scope::Create();
    NetBuilder net_builder("net_builder");
    std::unordered_map<std::string, Variable> var_map;
    std::unordered_map<std::string, std::string> var_model_to_program_map;

    SetInputToScope(input_data, input_types, input_shapes, target, ctx_scope);

    for (const auto& input_data_pair : input_data) {
      const std::string& input_name       = input_data_pair.first;
      const Type& input_type              = input_types.at(input_name);
      const std::vector<int>& input_shape = input_shapes.at(input_name);

      net_builder.CreateInput(input_type, input_shape, input_name);
      var_model_to_program_map[input_name] = input_name;
    }
    OpMapperContext op_mapper_ctx(*ctx_scope, target, &net_builder, &var_map, &var_model_to_program_map);
    const OpMapper* op_mapper = OpMapperRegistry::Global()->Find(op_desc.Type());
    CheckOpMapperNotNull(op_mapper);
    op_mapper->Run(op_desc, op_mapper_ctx);

    Program program                              = net_builder.Build();
    std::unordered_set<std::string> output_names = GetOutputNames(program);

    auto graph                       = std::make_shared<Graph>(program, target);
    std::shared_ptr<Scope> run_scope = BuildScope(target, graph);
    GraphCompiler gc(target, run_scope, graph);
    auto runtime_prog = gc.Build();

    SetInputToScope(input_data, input_types, input_shapes, target, run_scope);
    runtime_prog->Execute();
    return GetOutputFromScope(output_names, run_scope);
  }

  static std::unordered_map<std::string, std::vector<float>> GetModelToProgResult(
      const OpDesc& op_desc,
      const std::unordered_map<std::string, void*>& input_data,
      const std::unordered_map<std::string, Type>& input_types,
      const std::unordered_map<std::string, std::vector<int>>& input_shapes) {
#ifdef CINN_WITH_CUDA
    Target target = common::DefaultNVGPUTarget();
#else
    Target target = common::DefaultHostTarget();
#endif
    std::shared_ptr<Scope> ctx_scope = Scope::Create();
    SetInputToScope(input_data, input_types, input_shapes, target, ctx_scope);

    PaddleModelToProgram model_to_prog(ctx_scope.get(), target);
    model_to_prog.AddOp(op_desc);
    std::unique_ptr<Program> program = model_to_prog.GetProgram();

    for (size_t i = 0; i < program->size(); ++i) {
      for (Variable& x : (*program)[i]->inputs) {
        auto iter = input_types.find(x->id);
        if (iter != input_types.end()) {
          x->type  = iter->second;
          x->shape = input_shapes.at(x->id);
        }
      }
    }

    std::unordered_set<std::string> output_names = GetOutputNames(*program);
    auto graph                                   = std::make_shared<Graph>(*program, target);
    hlir::framework::ApplyPass(graph.get(), "InferShape");
    std::shared_ptr<Scope> run_scope = BuildScope(target, graph);
    GraphCompiler gc(target, run_scope, graph);
    auto runtime_prog = gc.Build();

    SetInputToScope(input_data, input_types, input_shapes, target, run_scope);
    runtime_prog->Execute();
    return GetOutputFromScope(output_names, run_scope);
  }
};

TEST(OpMapperRegistryTest, basic) {
  auto kernel = OpMapperRegistry::Global()->Find("sigmoid");
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(typeid(*kernel), typeid(OpMapper));
  ASSERT_EQ(kernel->name, "sigmoid");
}

// Test that the reverse of HW gets same result
// between OpMapper and PaddleModelToProgram
TEST(OpMapperRegistryTest, conv2d_reverse) {
  std::unique_ptr<OpDesc> op_desc = std::make_unique<OpDesc>();
  op_desc->SetType("conv2d");
  op_desc->SetInput("Input", {"input"});
  op_desc->SetInput("Filter", {"filter"});
  op_desc->SetOutput("Output", {"output"});
  op_desc->SetAttr<std::vector<int>>("paddings", {0, 0});
  op_desc->SetAttr<std::vector<int>>("strides", {1, 1});
  op_desc->SetAttr<std::vector<int>>("dilations", {1, 1});
  op_desc->SetAttr<int>("groups", 1);
  op_desc->SetAttr<std::string>("data_format", "AnyLayout");
  op_desc->SetAttr<std::string>("padding_algorithm", "EXPLICIT");

  std::unordered_map<std::string, Type> input_types(
      {{"input", ::cinn::common::F32()}, {"filter", ::cinn::common::F32()}});
  std::unordered_map<std::string, std::vector<int>> input_shapes({{"input", {1, 4, 5, 5}}, {"filter", {1, 4, 3, 3}}});

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int input_buffer_size = 1 * 4 * 5 * 5;
  float* input_buffer   = new float[input_buffer_size];
  for (int i = 0; i < input_buffer_size; ++i) {
    input_buffer[i] = dist(engine);
  }
  int filter_buffer_size = 1 * 4 * 3 * 3;
  float* filter_buffer   = new float[filter_buffer_size];
  for (int i = 0; i < filter_buffer_size; ++i) {
    filter_buffer[i] = dist(engine);
  }
  std::unordered_map<std::string, void*> input_data;
  input_data["input"]  = (void*)input_buffer;
  input_data["filter"] = (void*)filter_buffer;

  OpMapperTestUtil::AssertOpMapReg_Same_ModelToProg(*op_desc, input_data, input_types, input_shapes);
  delete[] input_buffer;
  delete[] filter_buffer;
}

}  // namespace frontend
}  // namespace cinn
