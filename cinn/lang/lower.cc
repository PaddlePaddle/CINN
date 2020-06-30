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
std::vector<ir::Buffer> GetTempBuffers(const std::vector<Tensor>& tensor_args, Expr body) {
  std::unordered_set<std::string> tensor_arg_names;

  for (auto& tensor : tensor_args) {
    tensor_arg_names.insert(tensor->name);
  }

  std::unordered_set<std::string> temp_buffer_names;  // used to avoid duplication.
  std::vector<ir::Buffer> temp_buffers;
  auto all_tensors = ir::CollectIRNodes(
      body, [&](const Expr* x) { return x->as_tensor() && !tensor_arg_names.count(x->as_tensor()->name); });
  for (auto& e : all_tensors) {
    if (!temp_buffer_names.count(e.as_tensor()->buffer->name)) {
      temp_buffers.push_back(e.as_tensor()->buffer);
      temp_buffer_names.insert(e.as_tensor()->buffer->name);
    }
  }
  return temp_buffers;
}

ir::LoweredFunc Lower(const std::string& name,
                      const std::vector<Tensor>& tensor_args,
                      const std::vector<Var>& scalar_args,
                      const std::vector<Tensor>& temp_tensors,
                      Module::Builder* b) {
  bool contains_gpu = false;
  for (auto& t : tensor_args) {
    if (contains_gpu = detail::TensorContainsGPUInfo(t)) break;
  }

  auto lower_impl_instance = detail::LowerImpl(name, tensor_args, scalar_args, temp_tensors);

  auto res = lower_impl_instance();

  if (b) {
    for (auto& temp_buffer : GetTempBuffers(tensor_args, res->body)) {
      b->AddBuffer(temp_buffer);
    }
  }

  if (contains_gpu) {
    res->device_api = ir::DeviceAPI::GPU;
  }

  if (b) {
    b->AddFunction(res);
  }
  return res;
}

}  // namespace lang
}  // namespace cinn
