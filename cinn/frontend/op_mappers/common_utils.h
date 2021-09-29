#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/common.h"
#include "cinn/common/type.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace frontend {
namespace utils {

inline void MoveData(float* data, int i, int M, int N) {
  float temp = data[i];
  int cur    = i;  // current data index
  int pre    = (cur % M) * N + cur / M;
  while (pre != i) {
    data[cur] = data[pre];
    cur       = pre;
    pre       = (cur % M) * N + cur / M;
  }
  data[cur] = temp;
}

inline void TransposeData(float* data, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    int next = (i % N) * M + i / N;
    while (next > i)  // next < 1 implies duplicate
      next = (next % N) * M + next / N;
    if (next == i)  // process current ring
      MoveData(data, i, M, N);
  }
}

inline void AddVar(const std::string& name, const Variable& var, const OpMapperContext& ctx, bool replace = false) {
  CheckVarNameValid(name);
  if (replace == false) {
    CHECK(!ctx.var_map_->count(name)) << "Duplicate variable [" << name << "] found";
  }
  (*ctx.var_map_)[name] = var;
}

inline Variable GetVar(const std::string& name, const OpMapperContext& ctx) {
  CheckVarNameValid(name);

  auto it = ctx.var_map_->find(name);
  if (it != ctx.var_map_->end()) return it->second;

  auto* var = ctx.scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    Variable var;
    var.set_id(name);
    var->shape = tensor->shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    AddVar(name, var, ctx);
    return var;
  }

  LOG(FATAL) << "No var called [" << name << "] exists";
  return Variable();
}

template <typename T>
inline T GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc, const std::string& name, const T& default_value = T{}) {
  if (op_desc.HasAttr(name)) {
    return op_desc.GetAttr<T>(name);
  }
  return default_value;
}

}  // namespace utils
}  // namespace frontend
}  // namespace cinn
