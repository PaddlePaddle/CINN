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

#include "cinn/hlir/framework/graph_compiler.h"

#include <absl/container/flat_hash_map.h>

#include <unordered_set>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/lang/lower.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace framework {
// Store params from node to instruction
void AddAttrs(const absl::flat_hash_map<std::string, AttrType>& attrs_store,
              const std::vector<std::string>& attrs_name,
              Instruction* instr) {
  for (auto& attr : attrs_name) {
    if (attrs_store.find(attr) != attrs_store.end()) {
      switch (attrs_store.at(attr).index()) {
        case 2:
          instr->attrs.push_back(absl::get<int>(attrs_store.at(attr)));
          break;
        case 3:
          instr->str_attrs.push_back(absl::get<std::string>(attrs_store.at(attr)));
          break;
        case 5:
          auto temp = absl::get<std::vector<int>>(attrs_store.at(attr));
          instr->attrs.insert(instr->attrs.end(), temp.begin(), temp.end());
          break;
      }
    } else {
      LOG(ERROR) << "Param " << attr << " missed! Please check.";
    }
  }
}

void GraphCompiler::PrintFunc() {
  auto topo_order = graph_->topological_order();
  auto& nodes     = std::get<0>(topo_order);
  auto& edges     = std::get<1>(topo_order);

  for (auto& n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
    }
  }
}

std::string GraphCompiler::GenSourceCode() {
  auto topo_order = graph_->topological_order();
  auto& nodes     = std::get<0>(topo_order);
  auto& edges     = std::get<1>(topo_order);

  for (auto& n : nodes) {
    auto* node = n->safe_as<Node>();
    if (node) {
      auto lowered_func = GetOpFunc(node);
      for (auto& i : lowered_func) {
        m_builder_.AddFunction(i);
      }
    }
  }
  // // compile the module
  if (!compiler_) {
    compiler_ = backends::Compiler::Create(target_);
  }

  auto build_module = m_builder_.Build();

  return compiler_->GetSourceCode(build_module);
}

std::vector<ir::LoweredFunc> GraphCompiler::GetOpFunc(const Node* node) {
  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  std::vector<std::vector<int>> output_shapes;
  VLOG(3) << "GetOpFunc of op " << node->id();
  for (auto& i : node->inlinks_in_order(true)) {
    std::string input_id = i->source()->as<NodeData>()->id();
    auto in_shape        = shape_dict.at(input_id);
    Type dtype           = dtype_dict.at(input_id);
    CHECK(dtype == Float(32) || dtype.is_bool() || dtype == Int(32))
        << "The dtype of node " << input_id << " is not float or bool or int! Other dtype is not implemented yet.";
    ir::Tensor temp;
    if (dtype == Float(32)) {
      temp = lang::Placeholder<float>(input_id, in_shape);
    } else if (dtype.is_bool()) {
      temp = lang::Placeholder<bool>(input_id, in_shape);
    } else if (dtype == Int(32)) {
      temp = lang::Placeholder<int>(input_id, in_shape);
    }
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }
  std::vector<Type> out_types;
  for (auto& out : node->outlinks_in_order(true)) {
    std::string out_id = out->sink()->safe_as<NodeData>()->id();
    auto out_shape     = shape_dict.at(out_id);
    Type dtype         = dtype_dict.at(out_id);
    output_shapes.push_back(out_shape);
    out_types.push_back(dtype);
  }
  auto impl = OpStrategy::SelectImpl(strategy[node->op()](node->attrs, inputs, out_types, output_shapes, target_));

  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
  poly::StageMap stages   = C.back();
  // make sure all the tensors in the stages before schedule launch.
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    stages->InsertLazily(temp.as_tensor_ref());
  }

  C = impl->fschedule(C);
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }

  auto func = lang::LowerVec(GenOpFuncName(node), stages, inputs, {}, {}, nullptr, this->target_);
  VLOG(3) << "The [" << func.size() << "] functions of node [" << node->attrs.node_name << "] are:\n";
  for (auto& i : func) {
    VLOG(3) << i;
  }
  return func;
}

// get the most complex op's index in the fused groups according to the OpPattern. If the OpPattern is same, we will
// take the latter.
int GetMasterRefNode(const std::vector<Node*>& nodes) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  int master_index      = 0;
  int master_pattern    = op_pattern_dict[nodes[0]->op()];
  for (int i = 1; i < nodes.size(); i++) {
    int pattern  = op_pattern_dict[nodes[i]->op()];
    master_index = pattern >= master_pattern ? i : master_index;
  }
  VLOG(3) << "master_index: " << master_index << ", master op: " << nodes[master_index]->op()->name;
  return master_index;
}

std::vector<ir::LoweredFunc> GraphCompiler::GetOpFunc(const std::vector<Node*>& nodes) {
  CHECK_GT(nodes.size(), 1) << "fuse nodes number must be greater than 1";
  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  int fuse_number  = nodes.size();
  VLOG(3) << "fuse begin: " << nodes[0]->id();
  std::vector<ir::Tensor> inputs;
  std::vector<ir::Tensor> outputs;
  poly::StageMap stages;
  int index             = 0;
  std::string fuse_name = "fn_";
  std::unordered_set<NodeData*> in_vars;
  std::unordered_set<NodeData*> out_vars;
  absl::flat_hash_map<NodeData*, Expr> temp_var_map;
  ir::Tensor master_out_tensor;
  int master_index = GetMasterRefNode(nodes);
  for (auto& node : nodes) {
    std::vector<ir::Tensor> temp_inputs;
    std::vector<common::CINNValue> cinn_inputs;
    std::vector<std::vector<int>> output_shapes;
    fuse_name += node->id() + "_";
    for (auto& link : node->inlinks_in_order(true)) {
      auto source = link->source();
      CHECK(source);
      auto source_data = source->as<NodeData>();
      CHECK(source_data);
      if (temp_var_map.count(source_data)) {
        VLOG(3) << "fuse var: " << source_data->id();
        Expr fuse_out = temp_var_map[source_data];
        cinn_inputs.push_back(common::CINNValue(fuse_out));
        temp_inputs.push_back(fuse_out.as_tensor_ref());
      } else {
        std::string input_id = source_data->id();
        auto in_shape        = shape_dict.at(input_id);
        Type dtype           = dtype_dict.at(input_id);
        CHECK(dtype == Float(32) || dtype.is_bool() || dtype == Int(32))
            << "The dtype of node " << input_id << " is not float or bool or int! Other dtype is not implemented yet.";
        ir::Tensor temp_in;
        if (dtype == Float(32)) {
          temp_in = lang::Placeholder<float>(input_id, in_shape);
        } else if (dtype.is_bool()) {
          temp_in = lang::Placeholder<bool>(input_id, in_shape);
        } else if (dtype == Int(32)) {
          temp_in = lang::Placeholder<int>(input_id, in_shape);
        }
        inputs.push_back(temp_in);
        temp_inputs.push_back(temp_in);
        cinn_inputs.push_back(common::CINNValue(temp_in));
        temp_var_map[source_data] = Expr(temp_in);
      }
      in_vars.insert(source_data);
    }
    std::vector<Type> out_types;
    std::vector<NodeData*> temp_outvars;
    for (auto& out : node->outlinks_in_order(true)) {
      auto out_var = out->sink()->safe_as<NodeData>();
      CHECK(out_var);
      out_vars.insert(out_var);
      temp_outvars.push_back(out_var);
      std::string out_id = out_var->id();
      VLOG(3) << "out_id " << out_id;
      auto out_shape = shape_dict.at(out_id);
      Type dtype     = dtype_dict.at(out_id);
      output_shapes.push_back(out_shape);
      out_types.push_back(dtype);
    }
    auto impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, temp_inputs, out_types, output_shapes, target_));

    common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
    if (index == master_index) {
      // use the most complex op's schedule as the fused ops' schedule.
      C = impl->fschedule(C);
      CHECK(!C.empty());
      Expr out          = C[0];
      master_out_tensor = out.as_tensor_ref();
    }
    CHECK_GE(C.size(), 2);
    CHECK_LE(C.size() - 1, node->outlinks_in_order().size());
    for (int i = 0; i < C.size() - 1; i++) {
      temp_var_map[temp_outvars[i]] = C[i];
    }
    poly::StageMap temp_stages = C.back();

    for (auto& i : temp_stages) {
      auto tensor = ir::Tensor(i.second->tensor());
      stages->InsertLazily(tensor, i.second.get());
    }
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      stages->InsertLazily(temp.as_tensor_ref(), temp_stages[temp.as_tensor_ref()]);
      if (index < fuse_number - 1 && !temp.as_tensor_ref()->is_reduce_tensor()) {
        // assume that only the first out_var links to other op node which will compute inline
        if (i == 0) {
          VLOG(3) << "inline " << temp.as_tensor_ref()->name;
          stages[temp.as_tensor_ref()]->ComputeInline();
        } else {
          VLOG(3) << "add middle op's other out_vars: " << temp.as_tensor_ref()->name;
          outputs.push_back(temp.as_tensor_ref());
        }
      } else if (index < fuse_number - 1 && temp.as_tensor_ref()->is_reduce_tensor()) {
        VLOG(3) << "temp buffer " << temp.as_tensor_ref()->name;
        if (target_.arch == Target::Arch::X86) {
          CHECK_NE(i, 0);
          outputs.push_back(temp.as_tensor_ref());
        } else {
          temp.as_tensor_ref()->WithBuffer("local", "_" + temp.as_tensor_ref()->name + "_temp_buffer");
          stages[temp.as_tensor_ref()]->SetScope(poly::ScopeKind::kLocal);
        }
      } else {
        if (index == fuse_number - 1) {
          // final output tensor
          outputs.insert(outputs.begin(), temp.as_tensor_ref());
        } else {
          outputs.push_back(temp.as_tensor_ref());
        }
      }
    }
    index++;
  }
  fuse_name += "fused";
  VLOG(3) << "fuse_name: " << fuse_name;
  // args order: inputs + final output + other no_fused outputs
  inputs.insert(inputs.end(), outputs.begin(), outputs.end());

  ir::Tensor final_out_tensor = outputs.front();
  if (final_out_tensor->name != master_out_tensor->name) {
    stages[final_out_tensor]->CopyTransform(stages[master_out_tensor]);
    stages[final_out_tensor]->CopyLoopInfo(stages[master_out_tensor]);
  }

  for (auto& s : stages) {
    auto& compute_ats = s.second->GetComputeAts();
    auto tensor       = s.second->tensor();
    if (tensor->is_reduce_tensor() && !compute_ats.empty()) {
      poly::ComputeAtRelation new_relation;
      CHECK_EQ(compute_ats.size(), 1U);
      auto new_stage = stages[final_out_tensor];
      for (auto& compute_at : compute_ats) {
        auto& old_relation     = compute_at.second;
        auto old_target_tensor = old_relation.stage->tensor();
        if (stages[old_target_tensor]->inlined()) {
          new_relation.stage = new_stage;
          new_relation.level = old_relation.level;

          compute_ats.clear();
          CHECK(new_relation.IsCompatible(s.second.get())) << "new computeAt should be compatible";
          compute_ats[new_stage->id()] = new_relation;
          break;
        }
      }
    }
  }

  auto func = lang::LowerVec(fuse_name, stages, inputs, {}, {}, nullptr, this->target_);
  VLOG(3) << "The [" << func.size() << "] functions are:\n";
  for (auto& i : func) {
    VLOG(3) << "Function [" << i->name << "] is:\n";
    VLOG(3) << i;
  }
  return func;
}

void GraphCompiler::ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_func) {
  if (lowered_func.size() > 1) {
    for (auto& i : lowered_func) {
      VLOG(3) << "In lowered_func, its name is : " << i->name;
      std::vector<std::string> input_args;
      std::vector<std::string> output_args;
      for (auto& j : i->args) {
        std::string temp_arg = j.name();
        if (temp_arg[0] == '_') temp_arg = temp_arg.substr(1);
        if (j.io == ir::Argument::IO::kOutput)
          output_args.push_back(temp_arg);
        else if (j.io == ir::Argument::IO::kInput)
          input_args.push_back(temp_arg);
        auto* var = scope_->FindVar(temp_arg);
        // For tensor not in scope, create it.
        if (!var) {
          auto* new_var = scope_->Var<Tensor>(temp_arg);
          auto& tensor  = absl::get<Tensor>(*new_var);
          std::vector<Shape::dim_t> shape;
          CHECK(j.is_buffer());
          VLOG(3) << "Tensor " << temp_arg << " is not found in scope. Now create it with shape:";
          for (auto& shape_dim : j.buffer_arg()->shape) {
            VLOG(3) << shape_dim << ",";
            CHECK(shape_dim.is_constant());
            shape.push_back(static_cast<int>(shape_dim.get_constant()));
          }
          tensor->Resize(Shape{shape});
        }
      }
      function2input_args_[i->name]  = input_args;
      function2output_args_[i->name] = output_args;
      m_builder_.AddFunction(i);
    }
  } else {
    m_builder_.AddFunction(lowered_func[0]);
  }
}

std::unique_ptr<Program> GraphCompiler::Build(const std::string& code) {
  GraphCompiler::CompileOptions options;
  options.attached_code              = code;
  options.with_instantiate_variables = true;

  auto&& result = Build(options);
  return std::move(result.runtime_program);
}

GraphCompiler::CompilationResult GraphCompiler::Build(const GraphCompiler::CompileOptions& options) {
  auto topo_order = graph_->topological_order();
  auto& nodes     = std::get<0>(topo_order);
  auto& edges     = std::get<1>(topo_order);

  auto& groups = graph_->groups;

  if (!groups.empty()) {
    for (int i = 0; i < groups.size(); i++) {
      std::vector<ir::LoweredFunc> lowered_func;
      if (groups[i].size() == 1) {
        lowered_func = GetOpFunc(groups[i][0]);
      } else {
        lowered_func = GetOpFunc(groups[i]);
      }
      this->ProcessFunction(lowered_func);
    }
  } else {
    VLOG(3) << "not run opfusion pass";
    for (auto& node : nodes) {
      auto op_node = node->safe_as<Node>();
      if (op_node) {
        auto lowered_func = GetOpFunc(op_node);
        this->ProcessFunction(lowered_func);
        graph_->groups.push_back({op_node});
      }
    }
  }

  // compile the module
  if (!compiler_) {
    compiler_ = backends::Compiler::Create(target_);
  }

  auto build_module = m_builder_.Build();

  if (this->target_.arch == Target::Arch::X86) {
    CodeGenCX86 codegen(this->target_, CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(build_module, CodeGenC::OutputKind::CImpl);
    VLOG(3) << "[X86] C Code is:\n" << out;
  }

  compiler_->Build(build_module, options.attached_code);
  if (options.with_instantiate_variables) {
    VLOG(3) << "Initantiate all variables on compile-time";
    // All variables reside in scope_, so traverse it to instantiate each one
    for (auto& name : scope_->var_names()) {
      auto* var    = scope_->Var<Tensor>(std::string({name.data(), name.size()}));
      auto& tensor = absl::get<Tensor>(*var);
      tensor->mutable_data<float>(target_);
    }
  }

  GraphCompiler::CompilationResult result;
  result.runtime_program.reset(new Program(scope_, BuildInstructions()));
  return result;
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions() {
  std::vector<std::unique_ptr<Instruction>> instructions;
  auto topo_order = graph_->topological_order();
  auto& nodes     = std::get<0>(topo_order);
  auto& edges     = std::get<1>(topo_order);

  auto& groups = graph_->groups;
  for (auto& group : groups) {
    if (group.size() == 1) {
      auto node  = group[0];
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_, scope_.get(), OpGetInputNames(node), OpGetOutputNames(node), node->op()->name));
      if (target_.arch == Target::Arch::NVGPU) {
        if (node->op()->name == "conv2d") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          // paddings strides dilations  groups
          AddAttrs(node->attrs.attr_store, {"paddings", "strides", "dilations"}, instr.get());
          if (node->attrs.attr_store.find("groups") != node->attrs.attr_store.end()) {
            auto groups = absl::get<int>(node->attrs.attr_store.at("groups"));
            instr->attrs.push_back(groups);
          } else {
            instr->attrs.push_back(1);
          }
          // output shape
          CHECK(!node->outlinks_in_order().empty());
          auto& out_node     = node->outlinks_in_order().front();
          std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
          auto out_shape     = shape_dict.at(out_id);
          instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          CHECK_EQ(instr->attrs.size(), 19UL);
          // conv type {forward, backward_data, backward_filter}
          std::string type = "forward";
          if (node->attrs.attr_store.find("conv_type") != node->attrs.attr_store.end()) {
            type = absl::get<std::string>(node->attrs.attr_store.at("conv_type"));
          }
          instr->str_attrs.push_back(type);
        } else if (node->op()->name == "depthwise_conv2d") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          // conv
          AddAttrs(node->attrs.attr_store, {"padding", "stride", "dilation"}, instr.get());
          if (node->attrs.attr_store.find("groups") != node->attrs.attr_store.end()) {
            auto groups = absl::get<int>(node->attrs.attr_store.at("groups"));
            instr->attrs.push_back(groups);
          } else {
            instr->attrs.push_back(instr->attrs[1]);
          }
          // output shape
          CHECK(!node->outlinks_in_order().empty());
          auto& out_node     = node->outlinks_in_order().front();
          std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
          auto out_shape     = shape_dict.at(out_id);
          instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          CHECK_EQ(instr->attrs.size(), 19UL);
          // conv type {forward, backward_data, backward_filter}
          std::string type = "forward";
          if (node->attrs.attr_store.find("conv_type") != node->attrs.attr_store.end()) {
            type = absl::get<std::string>(node->attrs.attr_store.at("conv_type"));
          }
          instr->str_attrs.push_back(type);
        } else if (node->op()->name == "pool2d") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            CHECK_EQ(in_shape.size(), 4UL);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          bool global_pooling = false;
          if (node->attrs.attr_store.find("global_pooling") != node->attrs.attr_store.end()) {
            global_pooling = absl::get<bool>(node->attrs.attr_store.at("global_pooling"));
          }
          if (node->attrs.attr_store.find("kernel_size") != node->attrs.attr_store.end()) {
            if (global_pooling == false) {
              auto padding = absl::get<std::vector<int>>(node->attrs.attr_store.at("kernel_size"));
              instr->attrs.insert(instr->attrs.end(), padding.begin(), padding.end());
            } else {
              instr->attrs.push_back(instr->attrs[2]);
              instr->attrs.push_back(instr->attrs[3]);
            }
          }
          if (node->attrs.attr_store.find("padding_size") != node->attrs.attr_store.end()) {
            if (global_pooling == false) {
              auto stride = absl::get<std::vector<int>>(node->attrs.attr_store.at("padding_size"));
              CHECK_EQ(stride.size(), 4UL);
              instr->attrs.insert(instr->attrs.end(), stride.begin(), stride.end());
            } else {
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
              instr->attrs.push_back(0);
            }
          }
          AddAttrs(node->attrs.attr_store, {"stride_size", "pool_type"}, instr.get());

          for (auto& out_node : node->outlinks_in_order()) {
            std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
            auto out_shape     = shape_dict.at(out_id);
            instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
          }
          if (node->attrs.attr_store.find("adaptive") != node->attrs.attr_store.end()) {
            bool adaptive = absl::get<bool>(node->attrs.attr_store.at("adaptive"));
            if (adaptive)
              instr->attrs.push_back(1);
            else
              instr->attrs.push_back(0);
          }
          CHECK_EQ(instr->attrs.size(), 17UL);
          CHECK_EQ(instr->str_attrs.size(), 1UL);
        } else if (node->op()->name == "softmax") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          AddAttrs(node->attrs.attr_store, {"axis"}, instr.get());
        } else if (node->op()->name == "mul") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          if (node->attrs.attr_store.find("x_num_col_dims") != node->attrs.attr_store.end()) {
            auto axis = absl::get<int>(node->attrs.attr_store.at("x_num_col_dims"));
            instr->attrs.push_back(axis);
          } else {
            instr->attrs.push_back(1);
          }
          if (node->attrs.attr_store.find("y_num_col_dims") != node->attrs.attr_store.end()) {
            auto axis = absl::get<int>(node->attrs.attr_store.at("y_num_col_dims"));
            instr->attrs.push_back(axis);
          } else {
            instr->attrs.push_back(1);
          }
        }
      }
      std::string op_func_name = GenOpFuncName(node);
      auto* fn                 = compiler_->Lookup(op_func_name);
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      int i                   = 1;
      std::string new_op_func = op_func_name + "_" + std::to_string(i);
      if (function2input_args_.count(new_op_func) != 0) {
        CHECK_GT(function2input_args_.count(op_func_name), 0);
        instr->AddInArgs(function2input_args_[op_func_name]);
        instr->AddOutArgs(function2output_args_[op_func_name]);
      }
      while (function2input_args_.count(new_op_func) != 0) {
        auto* fn2 = compiler_->Lookup(new_op_func);
        CHECK(fn2);
        instr->SetLoweredFunc(fn2);
        instr->AddInArgs(function2input_args_[new_op_func]);
        instr->AddOutArgs(function2output_args_[new_op_func]);
        i++;
        new_op_func = op_func_name + "_" + std::to_string(i);
      }
      if (node->attrs.attr_store.count("pre_run")) {
        instr->pre_run = absl::get<bool>(node->attrs.attr_store["pre_run"]);
      }
      instructions.push_back(std::move(instr));
    } else {
      CHECK_GT(group.size(), 1U) << "fuse number should be greater than 1";
      std::vector<std::string> inputNames;
      std::vector<std::string> outputNames;
      std::unordered_set<std::string> names_set;
      int count             = 0;
      std::string fuse_name = "fn_";
      for (int i = 0; i < group.size(); i++) {
        auto node = group[i];
        CHECK(node);
        fuse_name += node->id() + "_";
        auto temp_inputnames = OpGetInputNames(node);
        for (int j = 0; j < temp_inputnames.size(); j++) {
          if (!names_set.count(temp_inputnames[j])) {
            inputNames.push_back(temp_inputnames[j]);
            names_set.insert(temp_inputnames[j]);
          }
        }
        auto temp_outputnames = OpGetOutputNames(node);
        // fused output arg order: final output, ops no_fused outputs
        for (int j = 0; j < temp_outputnames.size(); j++) {
          if (!names_set.count(temp_outputnames[j])) {
            names_set.insert(temp_outputnames[j]);
            // assume that the first out_var of the op node is the fused var
            if (j == 0 && i != group.size() - 1) continue;
            if (j == 0 && i == group.size() - 1) {
              outputNames.insert(outputNames.begin(), temp_outputnames[0]);
            } else {
              outputNames.push_back(temp_outputnames[j]);
            }
          }
        }
      }
      fuse_name += "fused";
      VLOG(3) << fuse_name;
      auto instr =
          std::unique_ptr<Instruction>(new Instruction(target_, scope_.get(), inputNames, outputNames, fuse_name));
      VLOG(3) << "input_names: " << utils::Join(inputNames, ", ");
      VLOG(3) << "out_names: " << utils::Join(outputNames, ", ");
      auto* fn = compiler_->Lookup(fuse_name);
      CHECK(fn);
      instr->SetLoweredFunc(fn);
      instructions.push_back(std::move(instr));
    }
  }
  return instructions;
}

std::vector<std::string> GraphCompiler::OpGetInputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->inlinks_in_order()) {
    res.push_back(i->source()->as<NodeData>()->id());
  }
  return res;
}

std::vector<std::string> GraphCompiler::OpGetOutputNames(const Node* node) const {
  std::vector<std::string> res;
  for (auto& i : node->outlinks_in_order()) {
    res.push_back(i->sink()->as<NodeData>()->id());
  }
  return res;
}

std::shared_ptr<Scope> BuildScope(Target target, const std::shared_ptr<Graph>& graph, std::shared_ptr<Scope> scope) {
  auto& shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  if (!scope) scope = std::make_shared<Scope>();
  for (auto& iter : shape_dict) {
    auto* var    = scope->Var<Tensor>(iter.first);
    auto& tensor = absl::get<Tensor>(*var);
    std::vector<Shape::dim_t> shape;
    for (auto& shape_dim : iter.second) {
      shape.push_back(Shape::dim_t(shape_dim));
    }
    VLOG(3) << "Tensor [" << iter.first << "] resize to " << utils::Join(shape, ",");
    tensor->Resize(Shape{shape});
    CHECK(dtype_dict.at(iter.first) == Float(32) || dtype_dict.at(iter.first).is_bool() ||
          dtype_dict.at(iter.first) == Int(32))
        << "The dtype of node " << iter.first << " is not float or bool or int! Other dtype is not implemented yet.";
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
