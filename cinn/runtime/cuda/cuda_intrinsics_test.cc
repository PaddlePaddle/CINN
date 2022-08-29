// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "cinn/backends/compiler.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/runtime/cuda/cuda_intrinsics.h"
#include "cinn/runtime/cuda/use_extern_funcs.h"

namespace cinn {
namespace runtime {
namespace cuda {

TEST(find_value_nd, basic) {
  Expr M(10), N(20);
  Placeholder<float> x("x", {M, N});
  auto y = Compute({N}, [&](Expr i) { return CallExtern("cinn_cuda_index_add", {x, M, x({Expr(5), Expr(3)}), i, N}); });

  auto stages = CreateStages({y});

  auto jit = backends::SimpleJIT::Create();

  ir::Module::Builder builder("module1", common::DefaultNVGPUTarget());

  auto fn = Lower("fn", stages, {x, y});
  LOG(INFO) << "fn:\n" << fn;

  builder.AddFunction(fn);

  jit->Link(builder.Build());

  auto fn_ptr = jit->Lookup("fn");
  auto fnp    = reinterpret_cast<lower_func_ptr_t>(fn_ptr);
  ASSERT_TRUE(fnp);

  auto* x_buf   = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()}).set_random().Build();
  auto* out_buf = common::BufferBuilder(Int(32), {N.as_int32()}).set_zero().Build();
  auto args     = common::ArgsBuilder().Add(x_buf).Add(out_buf).Build();
  fnp(args.data(), args.size());

  auto* x_buf_data   = reinterpret_cast<float*>(x_buf->memory);
  auto* out_buf_data = reinterpret_cast<int*>(out_buf->memory);

  for (int i = 0; i < out_buf->num_elements(); i++) {
    LOG_FIRST_N(INFO, 3) << out_buf_data[i];
    if (out_buf_data[i] != -1) {
      ASSERT_NEAR(x_buf_data[out_buf_data[i] * 20 + i], x_buf_data[5 * 20 + 3], 1e-5);
    }
  }
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
