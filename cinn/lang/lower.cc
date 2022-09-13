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

#include "cinn/lang/lower.h"

#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include <utility>

#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/lower_impl.h"
#include "cinn/optim/optimize.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace lang {

using ir::Tensor;
using poly::Stage;

std::vector<ir::Argument> GetArgs(const Expr& func_body, const std::vector<std::string>& input_output_nodes) {
  std::vector<ir::Argument> res;
  std::set<std::string> load_tensors;
  std::set<std::string> store_tensors;
  auto store_nodes = ir::CollectIRNodesWithoutTensor(func_body, [&](const Expr* x) {
    if (x->As<ir::Store>()) store_tensors.insert(x->As<ir::Store>()->tensor.as_tensor_ref()->name);
    return x->As<ir::Store>();
  });
  auto load_nodes  = ir::CollectIRNodesWithoutTensor(func_body, [&](const Expr* x) {
    if (x->As<ir::Load>()) load_tensors.insert(x->As<ir::Load>()->tensor.as_tensor_ref()->name);
    return x->As<ir::Load>();
  });

  for (auto& i : input_output_nodes) {
    if (load_tensors.count(i) && !store_tensors.count(i)) {
      for (auto& j : load_nodes) {
        auto load_tensor = j.As<ir::Load>()->tensor.as_tensor_ref();
        if (load_tensor->buffer.defined() && load_tensor->name == i) {
          res.emplace_back(load_tensor->buffer, ir::Argument::IO::kInput);
          break;
        }
      }
    } else if (store_tensors.count(i) && !load_tensors.count(i)) {
      for (auto& j : store_nodes) {
        auto store_tensor = j.As<ir::Store>()->tensor.as_tensor_ref();
        if (store_tensor->buffer.defined() && store_tensor->name == i) {
          res.emplace_back(store_tensor->buffer, ir::Argument::IO::kOutput);
          break;
        }
      }
    }
  }
  for (auto& i : input_output_nodes) VLOG(3) << "In input_output_nodes, arg has : " << i;
  for (auto& i : res) VLOG(3) << "In res, arg has : " << i.name();
  return res;
}

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<Tensor>& tensor_args,
                                       const poly::StageMap& stage_map,
                                       Expr body) {
  std::unordered_set<std::string> tensor_arg_names;
  std::unordered_set<std::string> buffer_arg_names;
  for (auto& tensor : tensor_args) {
    tensor_arg_names.insert(tensor->name);
    if (tensor->buffer.defined()) {
      buffer_arg_names.insert(tensor->buffer->name);
    }
  }
  std::map<std::string, ir::Buffer> name_to_buffer;  // used to avoid duplication.

  auto all_temp_tensors = ir::CollectIRNodesWithoutTensor(body, [&](const Expr* x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() &&
           (!stage_map->Lookup(x->as_tensor()->name) || !stage_map[x->as_tensor()]->inlined()) &&
           ((!buffer_arg_names.count(x->as_tensor()->buffer->name) && !tensor_arg_names.count(x->as_tensor()->name)) ||
            utils::Endswith(x->as_tensor()->buffer->name, "temp_buffer"));
  });
  for (auto& e : all_temp_tensors) {
    auto buffer_name = e.as_tensor()->buffer->name;
    if (!name_to_buffer.count(buffer_name)) {
      name_to_buffer[buffer_name] = e.as_tensor()->buffer;
    } else {
      if (e.as_tensor()->buffer->numel() < name_to_buffer[buffer_name]->numel()) {
        name_to_buffer[buffer_name] = e.as_tensor()->buffer;
      }
    }
  }
  std::vector<ir::Buffer> temp_buffers;
  for (auto& i : name_to_buffer) temp_buffers.push_back(i.second);
  return temp_buffers;
}

std::set<ir::Tensor> CollectTempTensorsFromCtrlDepends(StageMap stages, const std::vector<Tensor>& tensor_args) {
  std::set<ir::Tensor> res;
  for (auto& stage : stages) {
    res.emplace(ir::Tensor(stage.second->tensor()));
    res.insert(stage.second->ctrl_depends().begin(), stage.second->ctrl_depends().end());
  }
  for (auto& t : tensor_args) {
    if (res.count(t)) res.erase(t);
  }
  return res;
}

void InitReduceTensor(StageMap stages, const Tensor& tensor, const Target& target) {
  if (tensor->is_reduce_tensor() && !tensor->IsReduceInited(stages)) {
    tensor->InitReduction(stages, target);
  }
  auto uninited_reduce_tensors = ir::CollectIRNodes(tensor->body(), [&](const Expr* x) {
    return x && x->defined() && x->as_tensor() && x->as_tensor()->is_reduce_tensor() &&
           !x->as_tensor()->IsReduceInited(stages);
  });
  for (auto& t : uninited_reduce_tensors) {
    VLOG(3) << "Init reduce tensor: " << t.as_tensor()->name;
    t.as_tensor()->InitReduction(stages, target);
  }
}

ir::LoweredFunc Lower(const std::string& name,
                      StageMap stages,
                      const std::vector<Tensor>& tensor_args,
                      const std::vector<Var>& scalar_args,
                      const std::vector<Tensor>& temp_tensors,
                      Module::Builder* b,
                      const Target& target,
                      bool support_ir_schedule) {
  // Init the reduce tensors first before any process.
  for (auto& t : tensor_args) InitReduceTensor(stages, t, target);
  for (auto& t : temp_tensors) InitReduceTensor(stages, t, target);
  // Merge the ctrl_deps with the given temp_tensors ang get a new temp_tensors
  auto ctrl_deps = CollectTempTensorsFromCtrlDepends(stages, tensor_args);
  ctrl_deps.insert(temp_tensors.begin(), temp_tensors.end());
  auto lower_impl_instance = detail::LowerImpl(name,
                                               stages,
                                               tensor_args,
                                               scalar_args,
                                               std::vector<Tensor>(ctrl_deps.begin(), ctrl_deps.end()),
                                               target,
                                               support_ir_schedule);
  auto result              = lower_impl_instance();
  std::vector<ir::LoweredFunc> return_value;
  for (auto& res : result) {
    auto temp_buffers = GetTempBuffers(tensor_args, stages, res->body);
    if (b) {
      for (auto& temp_buffer : temp_buffers) {
        b->AddBuffer(temp_buffer);
      }
    }
    {
      for (auto& stage : stages) {
        if (stage.second->IfCudaBind()) {
          res->device_api = ir::DeviceAPI::GPU;
          break;
        }
      }
      if (target == common::DefaultNVGPUTarget()) {
        res->device_api = ir::DeviceAPI::GPU;
      }
    }
    if (b) {
      b->AddFunction(res);
    }
    res->temp_bufs = temp_buffers;
    return_value.push_back(res);
  }
  return return_value[0];
}

std::vector<ir::LoweredFunc> LowerVec(const std::string& name,
                                      StageMap stages,
                                      const std::vector<Tensor>& tensor_args,
                                      const std::vector<Var>& scalar_args,
                                      const std::vector<Tensor>& temp_tensors,
                                      Module::Builder* b,
                                      const Target& target,
                                      bool support_ir_schedule) {
  // Init the reduce tensors first before any process.
  for (auto& t : tensor_args) InitReduceTensor(stages, t, target);
  for (auto& t : temp_tensors) InitReduceTensor(stages, t, target);
  // Merge the ctrl_deps with the given temp_tensors ang get a new temp_tensors
  auto ctrl_deps = CollectTempTensorsFromCtrlDepends(stages, tensor_args);
  ctrl_deps.insert(temp_tensors.begin(), temp_tensors.end());
  auto lower_impl_instance = detail::LowerImpl(name,
                                               stages,
                                               tensor_args,
                                               scalar_args,
                                               std::vector<Tensor>(ctrl_deps.begin(), ctrl_deps.end()),
                                               target,
                                               support_ir_schedule);
  // return vectorof ir::LoweredFunc.
  auto result = lower_impl_instance();
  std::vector<ir::LoweredFunc> return_value;
  for (auto& res : result) {
    auto temp_buffers = GetTempBuffers(tensor_args, stages, res->body);
    if (b) {
      for (auto& temp_buffer : temp_buffers) {
        b->AddBuffer(temp_buffer);
      }
    }

    {  // set function device_api
      for (auto& stage : stages) {
        if (stage.second->IfCudaBind()) {
          res->device_api = ir::DeviceAPI::GPU;
          break;
        }
      }

      if (target == common::DefaultNVGPUTarget()) {
        res->device_api = ir::DeviceAPI::GPU;
      }
    }
    if (b) {
      b->AddFunction(res);
    }

    res->temp_bufs = temp_buffers;

    return_value.push_back(res);
  }
  return return_value;
}

}  // namespace lang
}  // namespace cinn
