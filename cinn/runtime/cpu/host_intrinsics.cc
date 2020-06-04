#include "cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/runtime/cpu/host_intrinsics.h"
#include "mkl_math.h"

extern "C" {

float cinn_cpu_exp_fp32(float a) { return std::exp(a); }
float cinn_cpu_ceil_fp32(float a) { return std::ceil(a); }
float cinn_cpu_floor_fp32(float a) { return std::floor(a); }
float cinn_cpu_tanh_fp32(float a) { return std::tanh(a); }

float __cinn_host_tanh_fp32(float x) { return std::tanh(x); }
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out) {
  CINN_CHECK_EQ(x->num_elements(), out->num_elements());
  int xn         = x->num_elements();
  auto* x_data   = (float*)(x->host_memory);
  auto* out_data = (float*)(out->host_memory);
  for (int i = 0; i < x->num_elements(); i++) {
    out_data[i] = __cinn_host_tanh_fp32(x_data[i]);
  }
}

float __cinn_host_ceil_fp32(float x) { return std::ceil(x); }
}

namespace cinn {
namespace runtime {
namespace cpu {
using backends::FunctionProto;

namespace {

bool RegisterRuntimeSymbols() {
  auto host_target = common::DefaultHostTarget();

  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_tanh_fp32, host_target, float, float);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_ceil_fp32, host_target, float, float);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_floor_fp32, host_target, float, float);
  REGISTER_EXTERN_FUNC_ONE_IN_ONE_OUT(cinn_cpu_exp_fp32, host_target, float, float);

  REGISTER_EXTERN_FUNC(__cinn_host_ceil_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .AddOutputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC(cinn_cpu_exp_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();
}

[[maybe_unused]] bool x = RegisterRuntimeSymbols();

}  // namespace
}  // namespace cpu
}  // namespace runtime
}  // namespace cinn
