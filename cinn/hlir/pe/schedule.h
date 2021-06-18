#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {
int GetInnerSplitter(int origin, int other_axis);

int SplitEven(int origin);

int GetBasicFactor(const Type &type, const common::Target &target);

int GetBetterSplitFactor(int shape, int split_factor);

int GetArrayPackingFactor(int shape, const Type &type, const common::Target &target);

void ScheduleInjectiveCPU(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target);

void MatmulScheduleCPU(poly::StageMap stage,
                       const ir::Tensor &output,
                       const ir::Tensor &packedB,
                       const common::Target &target);

void MulScheduleCPU(poly::StageMap stage,
                    const ir::Tensor &output,
                    const ir::Tensor &input_tensor,
                    const common::Target &target);

void GetConv2dFactors(std::unordered_map<std::string, int> *factors,
                      int oc,
                      int ic,
                      int ow,
                      const Type &type,
                      const common::Target &target);
void GetConv2d1x1Factors(std::unordered_map<std::string, int> *factors,
                         int oc,
                         int ic,
                         int oh,
                         int ow,
                         const Type &type,
                         const common::Target &target);

void Conv2d_NCHWc_Schedule_CPU(poly::StageMap stages,
                               const ir::Tensor &res,
                               ir::Tensor &packed_out,
                               const ir::Tensor &input_pad,
                               const ir::Tensor &weights_dilation,
                               const ir::Tensor &data,
                               const common::Target &target);

void Conv2d_NCHWc_Schedule_CPU_Nofuse(poly::StageMap stages,
                                      const ir::Tensor &res,
                                      ir::Tensor &packed_out,
                                      const ir::Tensor &input_pad,
                                      const ir::Tensor &weights_dilation,
                                      const ir::Tensor &data,
                                      const common::Target &target);

void Conv2d_NCHWc_1X1_Schedule_CPU(poly::StageMap stages,
                                   const ir::Tensor &res,
                                   ir::Tensor &packed_out,
                                   const ir::Tensor &input_pad,
                                   const ir::Tensor &weights_dilation,
                                   const ir::Tensor &data,
                                   const common::Target &target);

void Conv2d_NCHWc_1X1_Schedule_CPU_Nofuse(poly::StageMap stages,
                                          const ir::Tensor &res,
                                          ir::Tensor &packed_out,
                                          const ir::Tensor &input_pad,
                                          const ir::Tensor &weights_dilation,
                                          const ir::Tensor &data,
                                          const common::Target &target);

void CudaScheduleMul(poly::StageMap stages,
                     ir::Tensor output,
                     const std::vector<int> &output_shape,
                     const common::Target &target);

void CudaScheduleConv(poly::StageMap stages, ir::Tensor &input_pad, ir::Tensor &output, const common::Target &target);

void CudaScheduleInjective(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target);

void CudaSplitSchedule(poly::Stage *stage, const std::vector<int> &output_shape);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
