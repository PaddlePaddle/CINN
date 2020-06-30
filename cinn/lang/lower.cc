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

ir::LoweredFunc Lower(const std::string& name,
                      const std::vector<Tensor>& tensor_args,
                      const std::vector<Var>& scalar_args,
                      const std::vector<Tensor>& temp_tensors,
                      Module::Builder* b) {
  bool contains_gpu = false;
  for (auto& t : tensor_args) {
    if (contains_gpu = detail::TensorContainsGPUInfo(t)) break;
  }

  auto res = detail::LowerImpl(name, tensor_args, scalar_args, temp_tensors)();

  if (b) {
    for (auto& temp : temp_tensors) {
      if (temp->buffer.defined()) {
        b->AddBuffer(temp->buffer);
      }
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
