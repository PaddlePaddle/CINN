#include "cinn/hlir/op/contrib/squeeze.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/common/context.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {

TEST(Squeeze, SqueezeCase0) {
  int m = 128;
  int n = 32;
  int k = 32;
  Expr M(m), N(n), K(k);

  Placeholder<float> input("A", {1, 1, M, 1, N, K});
  std::vector<int> axis;

  auto output = hlir::pe::Squeeze(input.tensor(), axis);
  auto stages = CreateStages({input, output});
  auto func   = Lower("fn", stages, {input, output});
  LOG(INFO) << "func:\n" << func;

#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
  Module::Builder builder("Squeeze_Builder", target);
  builder.AddFunction(func);

  auto module                    = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);
  auto &host_module              = std::get<0>(host_module_device_module);
  auto &device_module            = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // nv jit compile to ptx
  backends::NVRTC_Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());
  // cuda_module load ptx
  runtime::cuda::CUDAModule cuda_module(ptx, runtime::cuda::CUDAModule::Kind::PTX);
#endif  // CINN_WITH_CUDA
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn