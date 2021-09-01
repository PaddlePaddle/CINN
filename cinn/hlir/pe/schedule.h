#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/hlir/pe/schedule_param.pb.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {
class ScheduleParam {
 public:
  ~ScheduleParam();
  ScheduleParam(const ScheduleParam &) = delete;
  ScheduleParam &operator=(const ScheduleParam &) = delete;
  static ScheduleParam &get_instance() {
    static ScheduleParam instance;
    return instance;
  }
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> &GetParam() { return param_data; }
  std::unordered_map<std::string, std::vector<int>> &operator[](const std::string &key) { return param_data[key]; }
  int Count(const std::string &key) { return param_data.count(key); }

 private:
  ScheduleParam();
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> param_data;
};

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

void PoolScheduleGPU(poly::StageMap stages, ir::Tensor &output, const common::Target &target);

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

void Depthwise_Conv2d_NCHWc_Schedule_CPU_Nofuse(poly::StageMap stages,
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

void CudaScheduleConv(poly::StageMap stages,
                      ir::Tensor &input_pad,
                      ir::Tensor &weights,
                      ir::Tensor &output,
                      const common::Target &target);

void CudaScheduleWinogradConv(poly::StageMap wino_stages,
                              ir::Tensor &wino_weights_dilation,
                              ir::Tensor &wino_input_pad,
                              ir::Tensor &wino_A,
                              ir::Tensor &wino_B,
                              ir::Tensor &wino_G,
                              ir::Tensor &kernel_pack,
                              ir::Tensor &input_tile,
                              ir::Tensor &data_pack,
                              ir::Tensor &bgemm,
                              ir::Tensor &inverse,
                              ir::Tensor &wino_conv,
                              const common::Target &target);

void CudaScheduleConv2(poly::StageMap stages,
                       ir::Tensor &input_pad,
                       ir::Tensor &weights,
                       ir::Tensor &output,
                       const common::Target &target,
                       const std::string &key);

void CudaScheduleInjective(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target);

void CudaSplitSchedule(poly::Stage *stage, const std::vector<int> &output_shape);

void CreateSerialData(const std::string &file_name = "default_serial.log");

void LoadSerialData(const std::string &file_name = "default_serial.log");

int GetMaxSplitter(int a, int b);
}  // namespace pe
}  // namespace hlir
}  // namespace cinn
