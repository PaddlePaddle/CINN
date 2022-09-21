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

#include <memory>
#include <unordered_set>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/common/context.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_lowering.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/transform_gpu_forloop.h"
#include "cinn/poly/stage.h"
DECLARE_bool(cinn_ir_schedule);
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

Program::Program(const std::shared_ptr<Scope>& scope, std::vector<std::unique_ptr<Instruction>>&& instrs)
    : scope_(scope) {
  for (auto& ins : instrs) {
    if (ins->pre_run) {
      prerun_instrs_.push_back(std::move(ins));
    } else {
      instrs_.push_back(std::move(ins));
    }
  }
}

void Program::PreRun(const std::map<std::string, cinn_pod_value_t>* name2podargs) {
  for (auto& ins : prerun_instrs_) {
    ins->Run(name2podargs);
  }
  for (auto& ins : instrs_) {
    if (ins->size() == 4) {
      ins->PreRun(name2podargs);
    }
  }
}

void Program::Export(const std::vector<std::string>& persistent_vars, const std::string& filename) {
  auto writeplaceholder = [=](int s, int n, FILE* f) -> int {
    int pos = ftell(f);
    for (int i = 0; i < s * n; i++) {
      fwrite("\0", 1, 1, f);
    }
    return pos;
  };
  auto setplaceholder = [=](int p, void* b, int s, int n, FILE* f) {
    int cur = ftell(f);
    fseek(f, p, SEEK_SET);
    fwrite(b, s, n, f);
    fseek(f, cur, SEEK_SET);
  };
  auto tellplaceholder = [=](int p, FILE* f) {
    int cur = ftell(f);
    setplaceholder(p, &cur, 4, 1, f);
  };
  auto padding = [=](int alignment, uint8_t value, FILE* f) {
    int cur     = ftell(f);
    int padding = (alignment - (cur % alignment)) % alignment;
    for (int i = 0; i < padding; i++) {
      fwrite(&value, 1, 1, f);
    }
  };
  auto varnames = scope_->var_names();
  std::unordered_map<std::string, int> varindex;
  for (int i = 0; i < varnames.size(); i++) {
    varindex[(std::string)varnames[i]] = i;
  }

  FILE* f = fopen(filename.c_str(), "w+");

  fwrite("CINN", 4, 1, f);
  int major_v = 0;
  int minor_v = 0;
  fwrite(&major_v, 4, 1, f);
  fwrite(&minor_v, 4, 1, f);
  int unused_v = 0;
  fwrite(&unused_v, 4, 1, f);

  // varname list
  int varnamesec = writeplaceholder(4, 1, f);
  int namesnum   = varnames.size();
  fwrite(&namesnum, 4, 1, f);
  int nameoffset = writeplaceholder(4, namesnum, f);
  for (int i = 0; i < namesnum; i++) {
    int namelen = varnames[i].size();
    fwrite(&namelen, 4, 1, f);
    tellplaceholder(nameoffset + i * 4, f);
    fwrite(varnames[i].data(), namelen, 1, f);
    fwrite("\0", 1, 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(varnamesec, f);
  // pod_values
  int buffersec = writeplaceholder(4, 1, f);
  int bufoffset = writeplaceholder(4, 1, f);
  padding(alignof(cinn_buffer_t), 0, f);
  tellplaceholder(bufoffset, f);
  std::vector<std::pair<cinn_buffer_t*, int>> pvars;
  for (auto& varname : varnames) {
    std::string name     = (std::string)varname;
    auto t               = scope_->GetTensor(name);
    cinn_buffer_t buffer = *t->buffer();
    buffer.memory        = (uint8_t*)0;
    if (std::find(persistent_vars.begin(), persistent_vars.end(), name) != persistent_vars.end()) {
      pvars.emplace_back(t->buffer(), ftell(f) + offsetof(cinn_buffer_t, memory));
    }
    fwrite(&buffer, sizeof(cinn_buffer_t), 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(buffersec, f);
  // persistent_buffers
  int pbuffer = writeplaceholder(4, 1, f);
  for (auto& p : pvars) {
    if (p.first->align) {
      padding(p.first->align, 0, f);
    }
    tellplaceholder(p.second, f);
    fwrite(p.first->memory, p.first->memory_size, 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(pbuffer, f);
  // instructions
  int instsec = writeplaceholder(4, 1, f);
  int insnum  = 0;
  for (auto& ins : instrs_) {
    ins->Run(nullptr, true);
    insnum += ins->GetFnNames().size();
  }
  fwrite(&insnum, 4, 1, f);
  int instplaceholder = writeplaceholder(4 * 3, insnum, f);
  int findex          = 0;
  for (auto& ins : instrs_) {
    auto in_args  = ins->GetInArgs();
    auto out_args = ins->GetOutArgs();
    auto fn_names = ins->GetFnNames();
    for (int i = 0; i < fn_names.size(); i++, findex++) {
      std::vector<std::string> all_args(in_args[i].begin(), in_args[i].end());
      all_args.insert(std::end(all_args), out_args[i].begin(), out_args[i].end());
      auto fname    = fn_names[i];
      int fnamesize = fname.size();
      fwrite(&fnamesize, 4, 1, f);
      tellplaceholder(instplaceholder + findex * 12, f);
      fwrite(fname.c_str(), fname.size(), 1, f);
      fwrite("\0", 1, 1, f);
      int argsize = all_args.size();
      setplaceholder(instplaceholder + findex * 12 + 4, &argsize, 4, 1, f);
      padding(alignof(cinn_pod_value_t), 0, f);
      tellplaceholder(instplaceholder + findex * 12 + 8, f);
      for (auto& arg : all_args) {
        uintptr_t bufindex = varindex[arg];
        cinn_pod_value_t v((cinn_buffer_t*)bufindex);
        fwrite(&v, sizeof(cinn_pod_value_t), 1, f);
      }
    }
  }
  padding(16, 0, f);
  tellplaceholder(instsec, f);
  fclose(f);
}

void Program::Execute(const std::map<std::string, cinn_pod_value_t>* name2podargs, void* stream, bool use_cache) {
  for (auto& ins : instrs_) {
    ins->Run(name2podargs, false, stream, use_cache);
  }
#ifdef CINN_WITH_CUDA
  VLOG(4) << "-- The value of the used stream: " << stream;
  if (instrs_[0]->target_.arch == Target::Arch::NVGPU && stream == nullptr) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
#endif
}

void Program::ExecuteTest(int repeat_) {
  cinn::utils::Timer timer1;
  for (int i = 0; i < 100; i++) {
    for (auto& ins : instrs_) {
      ins->Run();
    }
  }
  timer1.Start();
  for (int i = 0; i < repeat_; i++) {
    for (auto& ins : instrs_) {
      ins->Run();
    }
  }
#ifdef CINN_WITH_CUDA
  if (instrs_[0]->target_.arch == Target::Arch::NVGPU) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
#endif
  double test_op_time = timer1.Stop() / repeat_;
  VLOG(3) << "Repeat times: [" << repeat_ << "], average op time: [" << test_op_time << "] ms";
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

const std::string& GraphCompiler::GetOrGenFullFuncName(const std::string& prefix) {
  // try_emplace only insert once, so the same function
  // can get a consistent name next time
  prefix2full_namemap_.try_emplace(prefix, Context::Global().NewName(prefix));
  return prefix2full_namemap_.at(prefix);
}

std::vector<ir::LoweredFunc> GraphCompiler::GetOpFuncWithIRSchedule(
    const Node* node,
    const absl::flat_hash_map<std::string, Type>& type_dict_,
    const absl::flat_hash_map<std::string, shape_t>& shape_dict_) {
  // get input tensor and output tensor
  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<ir::Tensor> tensor_inputs;
  std::vector<common::CINNValue> cinn_inputs;
  std::vector<std::string> input_output_nodes;
  VLOG(3) << "GetOpFunc of op " << node->id();

  // 1.Collect inputs info and outputs info
  for (auto& i : node->inlinks_in_order(true)) {
    std::string id = i->source()->as<NodeData>()->id();
    auto shape     = shape_dict_.at(id);
    Type dtype     = type_dict_.at(id);
    CHECK(dtype.is_supported()) << "The dtype of node " << id
                                << " is not float or bool or int! Other dtype is not implemented yet.";
    ir::Tensor input;
    if (dtype == Float(32)) {
      input = lang::Placeholder<float>(id, shape);
    } else if (dtype.is_bool()) {
      input = lang::Placeholder<bool>(id, shape);
    } else if (dtype == Int(32)) {
      input = lang::Placeholder<int32_t>(id, shape);
    } else if (dtype == Int(64)) {
      input = lang::Placeholder<int64_t>(id, shape);
    }
    tensor_inputs.push_back(input);
    cinn_inputs.push_back(common::CINNValue(input));
    input_output_nodes.push_back(id);
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  auto node_datas = GetAllNodeData(node);
  for (auto node_data : node_datas) {
    // collect output node data name.
    std::string out_name = node_data->id();
    VLOG(3) << "cinn_inputs.push_back " << out_name;
    cinn_inputs.push_back(common::CINNValue(out_name));
    out_types.push_back(type_dict_.at(out_name));
    out_shapes.push_back(shape_dict_.at(out_name));
    input_output_nodes.push_back(out_name);
  }

  auto impl =
      OpStrategy::SelectImpl(cinn_strategy[node->op()](node->attrs, tensor_inputs, out_types, out_shapes, target_));

  auto res =
      GetFuncFromImpl(impl, common::CINNValuePack{cinn_inputs}, tensor_inputs, input_output_nodes, node->id(), target_);
  return res;
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
    CHECK(dtype.is_supported()) << "The dtype of node " << input_id
                                << " is not float or bool or int! Other dtype is not implemented yet.";
    ir::Tensor temp;
    if (dtype == Float(32)) {
      temp = lang::Placeholder<float>(input_id, in_shape);
    } else if (dtype.is_bool()) {
      temp = lang::Placeholder<bool>(input_id, in_shape);
    } else if (dtype == Int(32)) {
      temp = lang::Placeholder<int32_t>(input_id, in_shape);
    } else if (dtype == Int(64)) {
      temp = lang::Placeholder<int64_t>(input_id, in_shape);
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
    // checkout whether the tensor is with buffer.
    if ((!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) &&
        !stages[temp.as_tensor_ref()]->inlined()) {
      inputs.push_back(temp.as_tensor_ref());
    }
  }

  auto func = lang::LowerVec(GetOrGenFullFuncName(GenOpFuncName(node)), stages, inputs, {}, {}, nullptr, this->target_);
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
    int pattern    = op_pattern_dict[nodes[i]->op()];
    master_index   = pattern >= master_pattern ? i : master_index;
    master_pattern = std::max(pattern, master_pattern);
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
  absl::flat_hash_set<ir::Tensor> fetch_tensors;
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
        VLOG(3) << "duplicate var: " << source_data->id();
        Expr fuse_out = temp_var_map[source_data];
        cinn_inputs.push_back(common::CINNValue(fuse_out));
        temp_inputs.push_back(fuse_out.as_tensor_ref());
      } else {
        std::string input_id = source_data->id();
        auto in_shape        = shape_dict.at(input_id);
        Type dtype           = dtype_dict.at(input_id);
        CHECK(dtype.is_supported()) << "The dtype of node " << input_id
                                    << " is not float or bool or int! Other dtype is not implemented yet.";
        ir::Tensor temp_in;
        if (dtype == Float(32)) {
          temp_in = lang::Placeholder<float>(input_id, in_shape);
        } else if (dtype.is_bool()) {
          temp_in = lang::Placeholder<bool>(input_id, in_shape);
        } else if (dtype == Int(32)) {
          temp_in = lang::Placeholder<int32_t>(input_id, in_shape);
        } else if (dtype == Int(64)) {
          temp_in = lang::Placeholder<int64_t>(input_id, in_shape);
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
    std::vector<Expr> temp_C;
    if (C.size() - 1 > node->outlinks_in_order().size()) {
      for (int i = 1; i < C.size() - 1; i++) {
        ir::Expr temp = C[i];
        VLOG(1) << "C[" << i << "] name is : " << temp.as_tensor_ref()->name;
        outputs.push_back(temp.as_tensor_ref());
      }
      common::CINNValuePack C_temp{{C[0], C.back()}};
      C = C_temp;
    }
    for (int i = 0; i < C.size() - 1; i++) {
      Expr out                      = C[i];
      temp_var_map[temp_outvars[i]] = out;
      if (fetch_var_ids_.count(temp_outvars[i]->id())) {
        VLOG(3) << "get fetch output var " << temp_outvars[i]->id();
        CHECK(out.as_tensor());
        fetch_tensors.insert(out.as_tensor_ref());
      }
    }
    CHECK_LE(C.size() - 1, node->outlinks_in_order().size());
    poly::StageMap temp_stages = C.back();

    for (auto& i : temp_stages) {
      auto tensor = ir::Tensor(i.second->tensor());
      stages->InsertLazily(tensor, i.second.get());
    }
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      CHECK(temp.as_tensor());
      auto temp_tensor = temp.as_tensor_ref();
      stages->InsertLazily(temp_tensor, temp_stages[temp_tensor]);
      if (index < fuse_number - 1 && !temp_tensor->is_reduce_tensor()) {
        // assume that only the first out_var links to other op node which will compute inline
        if (fetch_tensors.count(temp_tensor)) {
          VLOG(3) << "add op's fetch out_vars: " << temp_tensor->name;
          outputs.insert(outputs.begin(), temp_tensor);
        } else if (i == 0) {
          VLOG(3) << "inline " << temp_tensor->name;
          stages[temp_tensor]->ComputeInline();
        } else {
          VLOG(3) << "add middle op's other out_vars: " << temp_tensor->name;
          outputs.push_back(temp_tensor);
        }
      } else if (index < fuse_number - 1 && temp_tensor->is_reduce_tensor()) {
        VLOG(3) << "temp buffer " << temp_tensor->name;
        VLOG(3) << "add op's out_vars: " << temp_tensor->name;
        outputs.push_back(temp_tensor);
      } else {
        if (index == fuse_number - 1) {
          // final output tensor
          outputs.insert(outputs.begin(), temp_tensor);
        } else {
          outputs.push_back(temp_tensor);
        }
      }
    }
    index++;
  }
  fuse_name += "fused";
  VLOG(3) << "fuse_name: " << fuse_name;
  // args order: inputs + final output + fetch outputs + other no_fused outputs
  for (auto& tensor : outputs) {
    // checkout the tensor is with buffer.
    if ((!tensor->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) && !stages[tensor]->inlined()) {
      inputs.push_back(tensor);
    }
  }

  ir::Tensor final_out_tensor = outputs.front();
  if (final_out_tensor->name != master_out_tensor->name) {
    if (final_out_tensor->is_reduce_tensor()) {
      VLOG(3) << "final_out_tensor is reduce tensor!";
    } else {
      stages[final_out_tensor]->CopyTransform(stages[master_out_tensor]);
      stages[final_out_tensor]->CopyLoopInfo(stages[master_out_tensor]);
    }
  }

  for (auto& s : stages) {
    auto& compute_ats = s.second->GetComputeAts();
    auto tensor       = s.second->tensor();
    if (!compute_ats.empty()) {
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
  // deal with fetch tensors, not compute_inline but do compute_at
  for (auto& fetch_tensor : fetch_tensors) {
    if (fetch_tensor->is_reduce_tensor() || fetch_tensor->name == final_out_tensor->name) continue;
    stages[fetch_tensor]->DisableComputeInline();
    int level = stages[final_out_tensor]->n_out_dims() - 1;
    VLOG(3) << "no fuse fetch tensor " << fetch_tensor->name << " and recomputeAt in level " << level;

    // if the fetch tensor size is 1, the fetch tensor cannot fuse by ComputeAt2
    int len = 1;
    for (const auto& dim : fetch_tensor->shape) {
      len *= dim.as_int32();
    }
    if (len <= 1) {
      continue;
    }

    stages[fetch_tensor]->ComputeAt2(stages[final_out_tensor], level);
  }

  auto func = lang::LowerVec(GetOrGenFullFuncName(fuse_name), stages, inputs, {}, {}, nullptr, this->target_);
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
          tensor->set_type(j.type());
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

void GraphCompiler::CompileOptions::Apply(const auto_schedule::TuningResult& tuning_result) {
  // joint all sub_graph into a whole graph
  for (auto&& sub_graph : tuning_result.tuned_graph) {
    groups.insert(groups.end(), sub_graph.groups.begin(), sub_graph.groups.end());
  }

  // joint all lowered_funcs together
  for (auto&& expr : tuning_result.optimized_exprs) {
    lowered_funcs.insert(lowered_funcs.end(), expr.lowered_funcs.begin(), expr.lowered_funcs.end());
  }
}

GraphCompiler::CompilationResult GraphCompiler::Build(const GraphCompiler::CompileOptions& options,
                                                      std::unordered_set<std::string>&& fetch_var_ids,
                                                      void* stream) {
  Context::Global().ResetNameId();
  compile_options_ = options;
  fetch_var_ids_   = std::move(fetch_var_ids);
  auto topo_order  = graph_->topological_order();
  auto& nodes      = std::get<0>(topo_order);
  VLOG(3) << "Begin GraphCompiler::Build";
  m_builder_.Clear();
  // if there are no avaiable groups, we will take each node as a group
  if (options.groups.empty() && graph_->groups.empty() && graph_->fusion_groups.empty()) {
    VLOG(3) << "not run opfusion pass";
    for (auto& node : nodes) {
      auto op_node = node->safe_as<Node>();
      if (op_node) {
        graph_->groups.push_back({op_node});
      }
    }
  }
  // use the input groups in options firstly if exists
  auto groups = options.groups.empty() ? graph_->groups : options.groups;

  // if the input lowered_funcs is empty, we will use the defalut lowering process to generate
  std::vector<std::vector<ir::LoweredFunc>> local_lowered_funcs;
  if (options.lowered_funcs.empty()) {
    // lowering of new fusion pass is not compatible with the groups from the input options,
    // thus process it seperately
    if (!graph_->fusion_groups.empty()) {
      auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
      auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

      OpLowerer op_lowerer(dtype_dict, shape_dict, target_);
      for (auto& group : graph_->fusion_groups) {
        VLOG(3) << "group_id is : " << group->group_id << ", and its number is : " << group->nodes.size();
        groups.push_back(std::move(group->CollectNodes()));
        // set node as output node from fetch_var_ids.
        for (auto node : groups.back()) {
          // get all node datas.
          for (auto& link : node->outlinks()) {
            auto node_data = link->sink()->safe_as<NodeData>();
            CHECK(node_data);
            // if node data is in fetch_var_ids.
            if (fetch_var_ids_.count(node_data->id())) {
              group->output_nodes.insert(node);
              break;
            }
          }
        }
        local_lowered_funcs.emplace_back(std::move(op_lowerer.Lower(group)));
        CHECK_EQ(local_lowered_funcs.back().size(), 1) << "Lowerd Function Is Not Equal 1!";
        VLOG(3) << local_lowered_funcs.back()[0];
      }
    } else {
      VLOG(3) << "fusion_groups is empty";
      std::vector<ir::LoweredFunc> lowered_func;
      if (FLAGS_cinn_ir_schedule) {
        auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
        auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
        for (int i = 0; i < groups.size(); i++) {
          for (int j = 0; j < groups[i].size(); j++) {
            lowered_func = GetOpFuncWithIRSchedule(groups[i][j], dtype_dict, shape_dict);
            local_lowered_funcs.emplace_back(std::move(lowered_func));
          }
        }
      } else {
        for (int i = 0; i < groups.size(); i++) {
          if (groups[i].size() == 1) {
            lowered_func = GetOpFunc(groups[i][0]);
          } else {
            lowered_func = GetOpFunc(groups[i]);
          }
          local_lowered_funcs.emplace_back(std::move(lowered_func));
        }
      }
    }
  }
  // use the input lowered_funcs in options firstly if exists
  const auto& lowered_funcs = options.lowered_funcs.empty() ? local_lowered_funcs : options.lowered_funcs;
  CHECK_EQ(groups.size(), lowered_funcs.size()) << "The size of groups and lowered_funcs shoule be equal";
  for (auto&& lowered_func : lowered_funcs) {
    this->ProcessFunction(lowered_func);
  }
  graph_->VisualizeGroupedGraph(groups, fetch_var_ids_);

  // compile the module
  // Need to create a new compiler for every call of Build,
  // because the underneath jit engine does't support addIRModule repeatedly now.
  compiler_ = backends::Compiler::Create(target_);

  auto build_module = m_builder_.Build();
  VLOG(3) << "End of m_builder_.Build()";
  if (this->target_.arch == Target::Arch::X86) {
    CodeGenCX86 codegen(this->target_, CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(build_module, CodeGenC::OutputKind::CImpl);
    VLOG(3) << "[X86] C Code is:\n" << out;
  }

  compiler_->Build(build_module, options.attached_code);
  VLOG(3) << "End of compiler_->Build";
  auto instructions = BuildInstructions(groups, graph_->fusion_groups);
  VLOG(3) << "End of BuildInstructions";
  if (options.remove_unused_variables) {
    RemoveInvalidVariables(instructions);
  }
  if (options.with_buffer_handle_instruction_inserted) {
    VLOG(3) << "option.with_buffer_handle_instruction_inserted enable";
    InsertBufferHandlers(&instructions);
  }

  if (options.with_instantiate_variables) {
    VLOG(3) << "Initantiate all variables on compile-time";
    // All variables reside in scope_, so traverse it to instantiate each one
    for (auto& name : scope_->var_names()) {
      auto* var    = scope_->Var<Tensor>(std::string({name.data(), name.size()}));
      auto& tensor = absl::get<Tensor>(*var);
      if (reuse_vars_map_.count(name)) {
        auto src_var_name = reuse_vars_map_.at(name);
        auto* src_var     = scope_->Var<Tensor>(src_var_name);
        auto& src_tensor  = absl::get<Tensor>(*src_var);
        tensor->set_buffer(src_tensor->get_buffer());
      } else {
        tensor->mutable_data(target_, tensor->type());
      }
    }
  }
  GraphCompiler::CompilationResult result;
  result.runtime_program.reset(new Program(scope_, std::move(instructions)));
  return result;
}

void GraphCompiler::SetSubKernels(Instruction* instr, const std::string& func_name) {
  int i                   = 1;
  std::string new_op_func = func_name + "_" + std::to_string(i);
  if (function2input_args_.count(new_op_func) != 0) {
    CHECK_GT(function2input_args_.count(func_name), 0);
    instr->AddInArgs(function2input_args_[func_name]);
    instr->AddOutArgs(function2output_args_[func_name]);
  }
  while (function2input_args_.count(new_op_func) != 0) {
    auto* fn_ptr = compiler_->Lookup(new_op_func);
    CHECK(fn_ptr);
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), new_op_func);
    instr->AddInArgs(function2input_args_[new_op_func]);
    instr->AddOutArgs(function2output_args_[new_op_func]);
    i++;
    new_op_func = func_name + "_" + std::to_string(i);
  }
}

void GraphCompiler::BuildCublasInstr(const Node& node, Instruction* instr) const {
  instr->ClearInArgs();
  instr->AddInArgs(OpGetInputNames(&node));
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  // shape info
  std::vector<int> shape_sizes;
  for (auto& in_node : node.inlinks_in_order()) {
    std::string in_id = in_node->source()->safe_as<NodeData>()->id();
    auto in_shape     = shape_dict.at(in_id);
    instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
    shape_sizes.push_back(in_shape.size());
  }
  // cublas_gemm has three input vars, and its output shape is equal to the input bias.
  // cublas_matmul only has two input vars, so we should get its output shape from shape_dict.
  if (node.op()->name == "cublas_matmul") {
    for (auto& out_node : node.outlinks_in_order()) {
      std::string out_id = out_node->sink()->safe_as<NodeData>()->id();
      auto out_shape     = shape_dict.at(out_id);
      instr->attrs.insert(instr->attrs.end(), out_shape.begin(), out_shape.end());
      shape_sizes.push_back(out_shape.size());
    }
  }
  instr->attrs.insert(instr->attrs.end(), shape_sizes.begin(), shape_sizes.end());
  // attribute info
  bool trans_a = false;
  if (node.attrs.attr_store.contains("trans_a")) {
    trans_a = absl::get<bool>(node.attrs.attr_store.at("trans_a"));
  }
  instr->attrs.push_back(static_cast<int>(trans_a));
  bool trans_b = false;
  if (node.attrs.attr_store.contains("trans_b")) {
    trans_b = absl::get<bool>(node.attrs.attr_store.at("trans_b"));
  }
  instr->attrs.push_back(static_cast<int>(trans_b));
  bool trans_out = false;
  if (node.attrs.attr_store.contains("trans_out")) {
    trans_out = absl::get<bool>(node.attrs.attr_store.at("trans_out"));
  }
  instr->attrs.push_back(static_cast<int>(trans_out));
  float alpha = 1.f;
  if (node.attrs.attr_store.contains("alpha")) {
    alpha = absl::get<float>(node.attrs.attr_store.at("alpha"));
  }
  instr->attrs.push_back(*reinterpret_cast<int*>(&alpha));
}

std::vector<std::unique_ptr<Instruction>> GraphCompiler::BuildInstructions(
    const std::vector<std::vector<Node*>>& groups, const std::vector<std::shared_ptr<Graph::Group>>& fusion_groups) {
  std::vector<std::unique_ptr<Instruction>> instructions;
  auto topo_order = graph_->topological_order();
  auto& nodes     = std::get<0>(topo_order);
  auto& edges     = std::get<1>(topo_order);
  VLOG(3) << "Begin GraphCompiler::BuildInstructions";
  CHECK_GT(groups.size(), 0);
  CHECK_EQ(fusion_groups.size() != 0, groups.size() == fusion_groups.size())
      << "fusion_groups's size must be 0 or equal to groups.";
  for (int idx = 0; idx < groups.size(); ++idx) {
    auto& group = groups[idx];
    std::shared_ptr<Graph::Group> fusion_group(nullptr);
    if (fusion_groups.size()) {
      fusion_group = fusion_groups[idx];
    }
    if (group.size() == 1) {
      auto node       = group[0];
      auto instr_name = node->op()->name;
      if (node->op()->name == "reshape" && compile_options_.with_instantiate_variables) {
        // not run instruction and shares buffer only when instantiate_variables
        auto& inlinks  = node->inlinks_in_order();
        auto& outlinks = node->outlinks_in_order();
        CHECK_EQ(inlinks.size(), 1U);
        CHECK_EQ(outlinks.size(), 1U);
        std::string in_id       = inlinks[0]->source()->safe_as<NodeData>()->id();
        std::string out_id      = outlinks[0]->sink()->safe_as<NodeData>()->id();
        reuse_vars_map_[out_id] = in_id;
        instr_name              = "no_run";
      }
      auto instr = std::unique_ptr<Instruction>(
          new Instruction(target_,
                          scope_.get(),
                          fusion_group.get() ? fusion_group->input_names : OpGetInputNames(node),
                          fusion_group.get() ? fusion_group->output_names : OpGetOutputNames(node),
                          instr_name));

      if (target_.arch == Target::Arch::NVGPU) {
        if (node->op()->name == "conv2d") {
          auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
          for (auto& in_node : node->inlinks_in_order()) {
            std::string in_id = in_node->source()->safe_as<NodeData>()->id();
            auto in_shape     = shape_dict.at(in_id);
            instr->attrs.insert(instr->attrs.end(), in_shape.begin(), in_shape.end());
          }
          // padding stride dilation  group
          AddAttrs(node->attrs.attr_store, {"padding", "stride", "dilation"}, instr.get());
          if (node->attrs.attr_store.find("groups") != node->attrs.attr_store.end()) {
            auto conv_groups = absl::get<int>(node->attrs.attr_store.at("groups"));
            instr->attrs.push_back(conv_groups);
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
        } else if (node->op()->name == "cublas_gemm" || node->op()->name == "cublas_matmul") {
          BuildCublasInstr(*node, instr.get());
        }
      }
      std::string op_func_name =
          fusion_group.get() ? fusion_group->GetFuncName() : GetOrGenFullFuncName(GenOpFuncName(node));
      auto* fn_ptr = compiler_->Lookup(op_func_name);
      CHECK(fn_ptr);
      instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), op_func_name);

      // As some instruction like reduce, will generate more than one kernel.
      // So try to find the rest kernel, if it exist.
      SetSubKernels(instr.get(), op_func_name);
      if (node->attrs.attr_store.count("pre_run")) {
        instr->pre_run = absl::get<bool>(node->attrs.attr_store["pre_run"]);
      }
      // explicitly call Finalize of the instruction after all assignments on it were done
      instr->Finalize();
      instructions.push_back(std::move(instr));
    } else {
      CHECK_GT(group.size(), 1U) << "fuse number should be greater than 1";
      std::vector<std::string> inputNames;
      std::vector<std::string> outputNames;
      std::unordered_set<std::string> names_set;
      int count             = 0;
      std::string fuse_name = "fn_";
      if (!fusion_group.get()) {
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
              bool is_fetch = fetch_var_ids_.count(temp_outputnames[j]);
              if (j == 0 && i != group.size() - 1 && !is_fetch) continue;
              if (j == 0 && i == group.size() - 1) {
                outputNames.insert(outputNames.begin(), temp_outputnames[0]);
              } else if (is_fetch) {
                VLOG(3) << "fetch var " << temp_outputnames[j];
                outputNames.insert(outputNames.begin(), temp_outputnames[j]);
              } else {
                outputNames.push_back(temp_outputnames[j]);
              }
            }
          }
        }

        fuse_name += "fused";
        VLOG(3) << "In buildInstructions, fuse_name is : " << fuse_name;
        VLOG(3) << "input_names: " << utils::Join(inputNames, ", ");
        VLOG(3) << "out_names: " << utils::Join(outputNames, ", ");
      }
      fuse_name = fusion_group.get() ? fusion_group->GetFuncName() : GetOrGenFullFuncName(fuse_name);
      auto instr =
          std::unique_ptr<Instruction>(new Instruction(target_,
                                                       scope_.get(),
                                                       fusion_group.get() ? fusion_group->input_names : inputNames,
                                                       fusion_group.get() ? fusion_group->output_names : outputNames,
                                                       fuse_name));

      auto* fn_ptr = compiler_->Lookup(fuse_name);
      CHECK(fn_ptr);
      instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), fuse_name);
      // As some situation like reduce,will generate more than one kernel.
      // So try to find the rest kernel, if it exist.
      SetSubKernels(instr.get(), fuse_name);

      for (int j = 0; j < group.size(); j++) {
        auto node = group[j];
        if (node->attrs.attr_store.count("pre_run") && absl::get<bool>(node->attrs.attr_store["pre_run"]) == true) {
          instr->pre_run = true;
        }
      }
      // explicitly call Finalize of the instruction after all assignments on it were done
      instr->Finalize();
      instructions.push_back(std::move(instr));
    }
  }
  return instructions;
}

void GraphCompiler::RemoveInvalidVariables(const std::vector<std::unique_ptr<Instruction>>& instructions) {
  // mark all variables are invalid initially
  std::unordered_set<std::string> invalid_variables;
  auto var_names = scope_->var_names();
  invalid_variables.reserve(var_names.size());
  std::transform(var_names.begin(),
                 var_names.end(),
                 std::inserter(invalid_variables, invalid_variables.end()),
                 [](const auto& name_view) { return std::string(name_view.data()); });

  // erase used variable names
  auto exclude_arguments_fn = [&invalid_variables](const std::vector<std::string>& args) {
    std::for_each(args.begin(), args.end(), [&invalid_variables](const std::string& var_name) {
      invalid_variables.erase(var_name);
    });
  };

  // iterate the arguments of each instruction, eliminate the
  // used variables, and remain variables are invalid finally
  auto unused_var_num = invalid_variables.size();
  VLOG(3) << "Before removing invalid variables: " << instructions.size() << " instructions, "
          << invalid_variables.size() << " variables";
  for (auto i = 0; i < instructions.size(); ++i) {
    const auto& instr    = instructions.at(i);
    const auto& in_args  = instr->GetInArgs();
    const auto& out_args = instr->GetOutArgs();
    std::for_each(in_args.begin(), in_args.end(), exclude_arguments_fn);
    std::for_each(out_args.begin(), out_args.end(), exclude_arguments_fn);

    VLOG(3) << "Instruction-" << i << " eliminate " << unused_var_num - invalid_variables.size() << " used variables";
    unused_var_num = invalid_variables.size();
  }

  VLOG(3) << "There are " << unused_var_num << " invalid variables to be removed from scope";
  std::for_each(invalid_variables.begin(), invalid_variables.end(), [this](const std::string& var_name) {
    scope_->EraseVar(var_name);
    VLOG(3) << "Variable(" << var_name << ") is erased";
  });
}

static void BufferMallocWithCallback(void* args, int num_args) {
  cinn_pod_value_t* pod_args = static_cast<cinn_pod_value_t*>(args);
  for (int i = 0; i < num_args; ++i) {
    cinn_buffer_t* buffer = static_cast<cinn_buffer_t*>(pod_args[i]);
    CHECK(buffer->external_malloc) << "external_malloc is nullptr";
    buffer->external_malloc->operator()(nullptr, buffer);
  }
}

static void BufferFreeWithCallback(void* args, int num_args) {
  cinn_pod_value_t* pod_args = static_cast<cinn_pod_value_t*>(args);
  for (int i = 0; i < num_args; ++i) {
    cinn_buffer_t* buffer = static_cast<cinn_buffer_t*>(pod_args[i]);
    CHECK(buffer->external_free) << "external_free is nullptr";
    buffer->external_free->operator()(nullptr, buffer);
  }
}

void GraphCompiler::AnalyzeVariableLifeTime(const std::vector<std::unique_ptr<Instruction>>& instructions,
                                            std::unordered_map<int, std::vector<std::string>>* step2malloc,
                                            std::unordered_map<int, std::vector<std::string>>* step2free) {
  absl::flat_hash_map<std::string, int> variable_last_used, variable_first_used;
  for (auto step = 0; step < instructions.size(); ++step) {
    const auto& instr = instructions.at(step);

    for (const auto& args : instr->GetInArgs()) {
      for (const auto& var_name : args) {
        // use try_emplace to record the first time a variable appearance
        variable_first_used.try_emplace(var_name, step);
        // will update until last time a variable used
        variable_last_used[var_name] = step;
      }
    }
    for (const auto& args : instr->GetOutArgs()) {
      for (const auto& var_name : args) {
        variable_first_used.try_emplace(var_name, step);
        variable_last_used[var_name] = step;
      }
    }
  }

  for (const auto& var2first : variable_first_used) {
    (*step2malloc)[var2first.second].emplace_back(var2first.first);
  }
  for (const auto& var2last : variable_last_used) {
    (*step2free)[var2last.second].emplace_back(var2last.first);
  }
}

void GraphCompiler::InsertBufferHandlers(std::vector<std::unique_ptr<Instruction>>* instructions) {
  std::unordered_map<int, std::vector<std::string>> step2malloc, step2free;
  AnalyzeVariableLifeTime(*instructions, &step2malloc, &step2free);

  std::vector<std::unique_ptr<Instruction>> results;
  for (auto step = 0; step < instructions->size(); ++step) {
    auto& instr = instructions->at(step);

    // insert a buffer malloc instruction applying on variables
    // before they are firstly used in the next instruction
    auto m_it = step2malloc.find(step);
    if (m_it != step2malloc.end()) {
      const auto& malloc_var_names = m_it->second;
      auto function_name           = "malloc_buffer_instruction_" + std::to_string(step);
      auto malloc_instr            = std::make_unique<Instruction>(
          common::DefaultHostTarget(), scope_.get(), malloc_var_names, std::vector<std::string>({}), function_name);
      malloc_instr->SetLoweredFunc(reinterpret_cast<void*>(BufferMallocWithCallback), function_name);
      malloc_instr->Finalize();
      results.emplace_back(std::move(malloc_instr));
    }

    // join the real computation instruction
    results.emplace_back(std::move(instr));

    // insert a buffer free instruction applying on variables
    // after no instruction will use them anymore
    auto f_it = step2free.find(step);
    if (f_it != step2free.end()) {
      const auto& free_var_names = f_it->second;
      auto function_name         = "free_buffer_instruction_" + std::to_string(step);
      auto free_instr            = std::make_unique<Instruction>(
          common::DefaultHostTarget(), scope_.get(), std::vector<std::string>({}), free_var_names, function_name);
      free_instr->SetLoweredFunc(reinterpret_cast<void*>(BufferFreeWithCallback), function_name);
      free_instr->Finalize();
      results.emplace_back(std::move(free_instr));
    }
  }

  // replace original instructions
  instructions->swap(results);
}

std::vector<std::string> GraphCompiler::OpGetInputNames(const Node* node) const {
  std::vector<std::string> res;
  if (node->op()->name == "cublas_gemm" || node->op()->name == "cublas_matmul" || node->op()->name == "conv2d" ||
      node->op()->name == "depthwise_conv2d" || node->op()->name == "pool2d" || node->op()->name == "softmax" ||
      node->op()->name == "mul" || node->op()->name == "matmul") {
    for (auto& i : node->inlinks_in_order()) {
      res.push_back(i->source()->as<NodeData>()->id());
    }
  } else {
    std::unordered_set<std::string> repeat;
    for (auto& inode : node->inlinks_in_order()) {
      auto id = inode->source()->as<NodeData>()->id();
      if (repeat.count(id)) {
        continue;
      }
      repeat.insert(id);
      res.push_back(id);
    }
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
    CHECK(dtype_dict.count(iter.first));
    CHECK(dtype_dict.at(iter.first) == Float(32) || dtype_dict.at(iter.first).is_bool() ||
          dtype_dict.at(iter.first) == Int(32) || dtype_dict.at(iter.first) == Int(64))
        << "The dtype of node " << iter.first << " is not float or bool or int! Its type "
        << dtype_dict.at(iter.first).type() << ", " << dtype_dict.at(iter.first).bits() << " is not implemented yet.";
    tensor->set_type(dtype_dict.at(iter.first));
  }
  return scope;
}

std::vector<std::vector<ir::LoweredFunc>> GraphCompiler::FusedGraphToLoweredFunc(
    const std::vector<std::vector<hlir::framework::Node*>>& graph) {
  std::vector<std::vector<ir::LoweredFunc>> result;
  for (const std::vector<hlir::framework::Node*>& node_group : graph) {
    CHECK(!node_group.empty()) << "Invalid graph which contains empty node group in GraphCompiler.";
    if (node_group.size() == 1) {
      result.emplace_back(NodeToLoweredFunc(*(node_group[0])));
    } else {
      result.emplace_back(FusedNodeGroupToLoweredFunc(node_group));
    }
  }
  return result;
}

std::vector<ir::LoweredFunc> GraphCompiler::NodeToLoweredFunc(const hlir::framework::Node& node) {
  VLOG(4) << "Start NodeToExpr of op " << node.id();
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");

  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_value_inputs;

  for (const common::Shared<common::GraphEdge>& i : node.inlinks_in_order(true)) {
    std::string input_id    = i->source()->as<NodeData>()->id();
    const shape_t& in_shape = shape_dict.at(input_id);
    Type in_dtype           = dtype_dict.at(input_id);

    ir::Tensor tmp = lang::CreatePlaceHolder(in_shape, in_dtype, input_id);
    inputs.push_back(tmp);
    cinn_value_inputs.push_back(common::CINNValue(tmp));
  }

  std::vector<Type> out_types;
  std::vector<std::vector<int>> output_shapes;
  for (const common::Shared<common::GraphEdge>& out : node.outlinks_in_order(true)) {
    std::string out_id = out->sink()->safe_as<NodeData>()->id();
    if (FLAGS_cinn_ir_schedule) {
      cinn_value_inputs.push_back(common::CINNValue(out_id));
    }
    const shape_t& out_shape = shape_dict.at(out_id);
    Type dtype               = dtype_dict.at(out_id);
    output_shapes.push_back(out_shape);
    out_types.push_back(dtype);
  }

  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");

  std::shared_ptr<OpImpl> impl =
      OpStrategy::SelectImpl(strategy[node.op()](node.attrs, inputs, out_types, output_shapes, target_));
  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_value_inputs});
  // The last element of the result of fcompute is a StageMap
  poly::StageMap stages = static_cast<poly::StageMap>(C.back());

  // make sure the outputs are also in the Lower inputs
  for (int i = 0; i < C->size() - 1; ++i) {
    ir::Expr temp = C[i];
    // checkout whether the tensor is with buffer.
    if ((!temp.as_tensor_ref()->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) &&
        !stages[temp.as_tensor_ref()]->inlined()) {
      // inputs is reused as args of LowerVec, so we add output Tensor here.
      inputs.push_back(temp.as_tensor_ref());
    }
  }
  // Note: we didn't call `C = impl->fschedule(C)` because we remove the pre-set
  // schedule in auto-tuning. Can we speed up tuning using pre-set? Consider it
  // in the future.

  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(GetOrGenFullFuncName(GenOpFuncName(&node)),
                                                      stages,
                                                      inputs,
                                                      {},
                                                      {},
                                                      nullptr,
                                                      target_,
                                                      /*support_ir_schedule*/ true);

  VLOG(6) << "The [" << funcs.size() << "] functions of node [" << node.attrs.node_name << "] are:\n";
  for (auto& f : funcs) {
    VLOG(6) << f;
  }

  return funcs;
}

std::vector<ir::LoweredFunc> GraphCompiler::FusedNodeGroupToLoweredFunc(
    const std::vector<hlir::framework::Node*>& node_group) {
  VLOG(4) << "fuse begin: " << node_group[0]->id();
  size_t fuse_number = node_group.size();
  CHECK_GT(fuse_number, 1UL) << "fuse nodes number must be greater than 1";

  auto& strategy   = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  auto& shape_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict = graph_->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");

  poly::StageMap stages;
  std::vector<ir::Tensor> inputs;
  std::vector<ir::Tensor> outputs;
  std::unordered_set<NodeData*> in_vars;
  std::unordered_set<NodeData*> out_vars;
  absl::flat_hash_map<NodeData*, Expr> temp_var_map;
  absl::flat_hash_set<ir::Tensor> fetch_tensors;
  std::string fuse_name = "fn_";
  int master_index      = GetMasterRefNode(node_group);
  for (size_t index = 0; index < node_group.size(); ++index) {
    const hlir::framework::Node* node = node_group[index];
    std::vector<ir::Tensor> temp_inputs;
    std::vector<common::CINNValue> cinn_value_inputs;

    fuse_name += node->id() + "_";
    for (const common::Shared<common::GraphEdge>& link : node->inlinks_in_order(true)) {
      NodeData* source_data = link->source()->as<NodeData>();
      in_vars.insert(source_data);
      if (temp_var_map.count(source_data)) {
        VLOG(6) << "duplicate var: " << source_data->id();
        Expr fuse_out = temp_var_map[source_data];
        cinn_value_inputs.push_back(common::CINNValue(fuse_out));
        temp_inputs.push_back(fuse_out.as_tensor_ref());
      } else {
        std::string input_id = source_data->id();
        shape_t in_shape     = shape_dict.at(input_id);
        Type in_dtype        = dtype_dict.at(input_id);
        ir::Tensor temp_in   = lang::CreatePlaceHolder(in_shape, in_dtype, input_id);
        inputs.push_back(temp_in);
        temp_inputs.push_back(temp_in);
        cinn_value_inputs.push_back(common::CINNValue(temp_in));
        temp_var_map[source_data] = Expr(temp_in);
      }
    }

    std::vector<std::vector<int>> output_shapes;
    std::vector<Type> out_types;
    std::vector<NodeData*> temp_outvars;
    for (const common::Shared<common::GraphEdge>& out : node->outlinks_in_order(true)) {
      NodeData* out_var = out->sink()->safe_as<NodeData>();
      out_vars.insert(out_var);

      temp_outvars.push_back(out_var);
      std::string out_id = out_var->id();
      if (FLAGS_cinn_ir_schedule) {
        cinn_value_inputs.push_back(common::CINNValue(out_id));
      }
      shape_t out_shape = shape_dict.at(out_id);
      Type dtype        = dtype_dict.at(out_id);
      output_shapes.push_back(out_shape);
      out_types.push_back(dtype);
    }

    std::shared_ptr<OpImpl> impl =
        OpStrategy::SelectImpl(strategy[node->op()](node->attrs, temp_inputs, out_types, output_shapes, target_));
    common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_value_inputs});

    // The last element of the result of fcompute is a StageMap
    poly::StageMap temp_stages = C.back();
    for (const auto& i : temp_stages) {
      ir::Tensor tensor(i.second->tensor());
      stages->InsertLazily(tensor, i.second.get());
    }

    for (int i = 0; i < C.size() - 1; i++) {
      Expr out                      = C[i];
      temp_var_map[temp_outvars[i]] = out;
      ir::Tensor temp_tensor        = out.as_tensor_ref();
      stages->InsertLazily(temp_tensor, temp_stages[temp_tensor]);
      if (fetch_var_ids_.count(temp_outvars[i]->id())) {
        VLOG(6) << "get fetch output var " << temp_outvars[i]->id();
        CHECK(out.as_tensor());
        fetch_tensors.insert(out.as_tensor_ref());
      }
      if (index == fuse_number - 1) {
        // final output tensor
        outputs.insert(outputs.begin(), temp_tensor);
      } else {
        outputs.push_back(temp_tensor);
      }
    }
  }
  fuse_name += "fused";
  VLOG(6) << "fuse_name: " << fuse_name;

  for (const ir::Tensor& tensor : outputs) {
    // checkout the tensor is with buffer.
    if (!tensor->buffer.defined() || this->target_ != common::DefaultNVGPUTarget()) {
      inputs.push_back(tensor);
    }
  }

  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      GetOrGenFullFuncName(fuse_name), stages, inputs, {}, {}, nullptr, this->target_, /*support_ir_schedule*/ true);

  VLOG(6) << "The [" << funcs.size() << "] functions of " << fuse_name << " are:\n";
  for (const ir::LoweredFunc& f : funcs) {
    VLOG(6) << "Function [" << f->name << "] is:\n";
    VLOG(6) << f;
  }

  return funcs;
}

std::vector<ir::LoweredFunc> GetFuncFromImpl(const std::shared_ptr<OpImpl>& impl,
                                             const common::CINNValuePack& cinn_inputs,
                                             std::vector<ir::Tensor>& all_arg_tensors,
                                             const std::vector<std::string>& input_output_nodes,
                                             const std::string& node_id,
                                             const Target& target) {
  // 1.Call Op's Compute function, using the default stages and LowerVec to get IR tree.
  common::CINNValuePack C = impl->fcompute(cinn_inputs);

  // 2. Collect tensors and arguments
  // Add output tensors to all_arg_tensors
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    // checkout whether the tensor is with buffer.
    if (!temp.as_tensor_ref()->buffer.defined() || target != common::DefaultNVGPUTarget()) {
      all_arg_tensors.push_back(temp.as_tensor_ref());
    }
  }

  poly::StageMap stages        = C.back();
  std::string func_name_prefix = "fn_";
  auto funcs = lang::LowerVec(func_name_prefix + node_id, stages, all_arg_tensors, {}, {}, nullptr, target, true);

  std::vector<common::CINNValue> schedule_inputs;
  for (int i = 0; i < C.size() - 1; ++i) {
    CHECK(C[i].is_tensor());
    schedule_inputs.push_back(common::CINNValue(C[i]));
  }
  for (auto& f : funcs) {
    schedule_inputs.push_back(common::CINNValue(f->body));
  }

  // 3. Call Op's Schedule function, optimizing the IR tree by new IR schedule
  common::CINNValuePack expr_pack = impl->fschedule(common::CINNValuePack{schedule_inputs});

  // 4. Optimize the LoweredFunc
  VLOG(3) << "expr_pack.size() is : " << expr_pack.size();
  std::vector<ir::LoweredFunc> res;
  for (int i = 0; i < expr_pack.size(); i++) {
    if (funcs.size() > expr_pack.size()) {
      auto new_args  = lang::GetArgs(funcs[i]->body, input_output_nodes);
      funcs[i]->args = new_args;
    }
    auto temp_buffers   = lang::GetTempBuffers(all_arg_tensors, stages, funcs[i]->body);
    funcs[i]->temp_bufs = temp_buffers;
    funcs[i]->PrepareBufferCastExprs();
    res.push_back(funcs[i]);
  }
  for (int i = 0; i < res.size(); i++) {
#ifdef CINN_WITH_CUDA
    optim::OptimizeExprGPU(&(res[i]->body));
#endif
    res[i] = optim::Optimize(Expr(res[i]), target, false).as_lowered_func_ref();
  }

  // 5. Return the result.
  return res;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
