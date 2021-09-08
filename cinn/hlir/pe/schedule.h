#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/hlir/framework/node.h"
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
  static ScheduleParam &get_cuda_instance() {
    static ScheduleParam cuda_instance;
    return cuda_instance;
  }
  static ScheduleParam &get_x86_instance() {
    static ScheduleParam x86_instance;
    return x86_instance;
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
                      int fc,
                      int oh,
                      int ow,
                      const Type &type,
                      const common::Target &target,
                      const std::string &key = "",
                      bool import_params     = true);

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
                               const common::Target &target,
                               const std::string &key,
                               bool do_padding);

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
                                   const common::Target &target,
                                   const std::string &key,
                                   bool do_padding);

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
                                                const common::Target &target,
                                                bool do_padding);

void CudaScheduleMul(poly::StageMap stages,
                     ir::Tensor output,
                     const std::vector<int> &output_shape,
                     const common::Target &target);

void CudaScheduleDepthwiseConv(poly::StageMap stages, ir::Tensor &output, const common::Target &target);

void CudaScheduleConv(poly::StageMap stages,
                      ir::Tensor &input_pad,
                      ir::Tensor &weights,
                      ir::Tensor &output,
                      const common::Target &target);

void CudaScheduleConv2(poly::StageMap stages,
                       ir::Tensor &input_pad,
                       ir::Tensor &weights,
                       ir::Tensor &output,
                       const common::Target &target,
                       const std::string &key);

void CudaScheduleInjective(poly::Stage *stage, const std::vector<int> &output_shape, const common::Target &target);

void CudaSplitSchedule(poly::Stage *stage, const std::vector<int> &output_shape);

void CreateCudaSerialData(const std::string &file_name = "default_serial.log");

std::string GenerateX86ConvKey(const std::vector<Expr> &input_shape,
                               const std::vector<Expr> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations);

std::string GenerateX86ConvKey(const std::vector<int> &input_shape,
                               const std::vector<int> &weight_shape,
                               const std::vector<int> &strides,
                               const std::vector<int> &paddings,
                               const std::vector<int> &dilations);
void CreateX86SerialData(const std::string &file_name = "default_serial.log");

void LoadSerialData(std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> *params,
                    const std::string &file_name = "default_serial.log");

void SaveSerialData(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>> &model_data,
    const std::string &file_name = "default_serial.log");

int GetMaxSplitter(int a, int b);
}  // namespace pe
}  // namespace hlir
}  // namespace cinn
