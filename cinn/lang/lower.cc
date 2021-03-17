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

namespace cinn {
namespace lang {

using ir::Tensor;
using poly::Stage;

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
  std::unordered_set<std::string> temp_buffer_names;  // used to avoid duplication.
  std::vector<ir::Buffer> temp_buffers;
  auto all_tensors = ir::CollectIRNodes(body, [&](const Expr* x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() && !stage_map[x->as_tensor()]->inlined() &&
           !buffer_arg_names.count(x->as_tensor()->buffer->name) && !tensor_arg_names.count(x->as_tensor()->name);
  });
  for (auto& e : all_tensors) {
    if (!temp_buffer_names.count(e.as_tensor()->buffer->name)) {
      temp_buffers.push_back(e.as_tensor()->buffer);
      temp_buffer_names.insert(e.as_tensor()->buffer->name);
    }
  }
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
                      const Target& target) {
  // Init the reduce tensors first before any process.
  for (auto& t : tensor_args) InitReduceTensor(stages, t, target);
  for (auto& t : temp_tensors) InitReduceTensor(stages, t, target);

  // Merge the ctrl_deps with the given temp_tensors ang get a new temp_tensors
  auto ctrl_deps = CollectTempTensorsFromCtrlDepends(stages, tensor_args);
  ctrl_deps.insert(temp_tensors.begin(), temp_tensors.end());

  auto lower_impl_instance = detail::LowerImpl(
      name, stages, tensor_args, scalar_args, std::vector<Tensor>(ctrl_deps.begin(), ctrl_deps.end()));

  auto res = lower_impl_instance();

  auto temp_buffers = GetTempBuffers(tensor_args, stages, res->body);
  if (b) {
    for (auto& temp_buffer : temp_buffers) {
      b->AddBuffer(temp_buffer);
    }
  }

  {  // set function device_api
    bool contains_gpu = false;
    for (auto& t : tensor_args) {
      if (contains_gpu = detail::TensorContainsGPUInfo(t, stages[t])) break;
    }
    if (!contains_gpu) {
      for (auto& t : temp_tensors) {
        if (contains_gpu = detail::TensorContainsGPUInfo(t, stages[t])) break;
      }
    }

    if (contains_gpu) {
      res->device_api = ir::DeviceAPI::GPU;
    }
  }

  if (b) {
    b->AddFunction(res);
  }

  res->temp_bufs = temp_buffers;

  return res;
}

}  // namespace lang
}  // namespace cinn
