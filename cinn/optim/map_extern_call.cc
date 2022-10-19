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

#include "cinn/optim/map_extern_call.h"

#include "cinn/cinn.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace optim {

static const std::set<std::string> kExternFp32CallsGPU{
    {"exp",   "erf",   "sigmoid", "sqrt",  "log",   "log2", "log10",    "floor", "ceil",
     "round", "trunc", "cos",     "cosh",  "tan",   "sin",  "sinh",     "acos",  "acosh",
     "asin",  "asinh", "atan",    "atanh", "isnan", "tanh", "isfinite", "isinf", "remainder"}};

static const std::set<std::string> kExternInt32CallsGPU{
    {"left_shift", "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"}};

static const std::set<std::string> kExternFp32CallsCPU = {
    "erf", "acos", "acosh", "asin", "asinh", "atan", "atanh", "remainder"};

void MapExternCall(Expr *e, Target target) {
  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    explicit Mutator(Target target) : target(target) {}

    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      CHECK(node);

      if (target.arch == Target::Arch::NVGPU) {
        DealWithNvGpuintrinsics(node, expr);
      } else {
        DealWithCpuintrinsics(node, expr);
      }
    }

    void DealWithCpuintrinsics(ir::Call *node, Expr *expr) {
      if (kExternFp32CallsCPU.count(node->name)) {
        CHECK_GE(node->read_args.size(), 1UL);
        CHECK_EQ(node->read_args.front().type(), Float(32));
        auto out_type = node->type();
        *expr         = lang::CallExtern(node->name + "f", node->read_args);
      }
    }

    void DealWithNvGpuintrinsics(ir::Call *node, Expr *expr) {
      auto arg_size = node->read_args.size();
      CHECK_GT(arg_size, 0UL) << "The arguments of " << node->name << " should not empty! Please check.";
      auto dtype = node->read_args.front().type();

      bool node_in_extern_fp32  = kExternFp32CallsGPU.count(node->name);
      bool node_in_extern_int32 = kExternInt32CallsGPU.count(node->name);

      if (node_in_extern_fp32 && node_in_extern_int32) {
        CHECK_GE(arg_size, 1UL);
        if (dtype == Float(32)) {
          *expr = lang::CallExtern("cinn_nvgpu_" + node->name + "_fp32", node->read_args);
        } else if (dtype == Int(32)) {
          *expr = lang::CallExtern("cinn_nvgpu_" + node->name + "_int32", node->read_args);
        } else {
          LOG(FATAL) << "Not support data type " << dtype;
        }
      } else if (node_in_extern_fp32) {
        CHECK_GE(arg_size, 1UL);
        CHECK_EQ(dtype, Float(32));
        *expr = lang::CallExtern("cinn_nvgpu_" + node->name + "_fp32", node->read_args);
      } else if (node_in_extern_int32) {
        CHECK_GE(arg_size, 1UL);
        CHECK_EQ(dtype, Int(32));
        *expr = lang::CallExtern("cinn_nvgpu_" + node->name + "_int32", node->read_args);
      }
      // TODO(Superjomn) deal with int64 intrinsics.
    }
  };

  Mutator m(target);
  m(e);
}

}  // namespace optim
}  // namespace cinn
