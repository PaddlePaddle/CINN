#include "cinn/hlir/instruction/x86/math_registors.h"

#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/function_prototype.h"
#include "cinn/hlir/instruction/x86/cpu_intrisics.h"
#include "cinn/hlir/instruction/x86/mkl_math.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace x86 {
using backends::FunctionProto;

bool RegisterMklMath() {
  auto host_target = common::DefaultHostTarget();

  REGISTER_EXTERN_FUNC(cinn_mkl_tanh_v_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC(cinn_mkl_tanh_v_fp64, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  REGISTER_EXTERN_FUNC(cinn_mkl_exp_v_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<cinn_buffer_t *>()
      .AddOutputType<cinn_buffer_t *>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

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
  REGISTER_EXTERN_FUNC(cinn_cpu_tanh_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();
  REGISTER_EXTERN_FUNC(cinn_cpu_ceil_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();
  REGISTER_EXTERN_FUNC(cinn_cpu_floor_fp32, host_target)
      .SetRetType<float>()
      .AddInputType<float>()
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))
      .End();

  return true;
}

}  // namespace x86
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
