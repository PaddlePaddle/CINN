#include "cinn/hlir/op/contrib/squeeze.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

ir::Tensor Squeeze(const ir::Tensor& A,
                   const std::vector<int>& axes,
                   poly::StageMap stages,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  std::vector<Expr> A_expr_shape = A->shape;
  if (axes.size()!=0){
    for (auto& a : axes) {
      CHECK(a<A_expr_shape.size());
      CHECK_EQ(A_expr_shape[a], Expr(1));
      A_expr_shape[a] = Expr(0);
    }
    for (auto& i : A_expr_shape) {
      CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
      if(i != Expr(0)){
        new_expr_shape.push_back(i);
      }
    }
  }else{
    for (auto& i : A_expr_shape) {
      CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
      if(i != Expr(1)){
        new_expr_shape.push_back(i);
      }
    }
  }

  auto out = Identity(A->Reshape(new_expr_shape, stages), name).front();
  return out;
}

ir::Tensor Squeeze(const ir::Tensor& A,
                   const std::vector<int>& axes,
                   const std::string& name) {
  std::vector<Expr> new_expr_shape;
  std::vector<Expr> A_expr_shape = A->shape;
  for (auto& i : A_expr_shape) {
    CHECK(i.is_constant()) << "Input tensor's shape should be constant value.";
    if(i != Expr(1)){
      new_expr_shape.push_back(i);
    }
  }
  auto res = Compute(
      new_expr_shape,
      [=](const std::vector<Expr>& indice) {
        Expr offset = Expr(0);
        for (int i = 0; i < indice.size(); i++) {
          offset = offset * new_expr_shape[i] + indice[i];
        }
        std::vector<Expr> indice_a;
        for (int i = A_expr_shape.size() - 1; i >= 0; i--) {
          auto temp = offset % A_expr_shape[i];
          indice_a.insert(indice_a.begin(), common::AutoSimplify(temp));
          offset = (offset - temp) / A_expr_shape[i];
        }
        return A(indice_a);
      },
      name);
  return res;
}


std::shared_ptr<OpStrategy> StrategyForSqueeze(const framework::NodeAttr &attrs,
                                               const std::vector<ir::Tensor> &inputs,
                                               const std::vector<Type> &out_type,
                                               const std::vector<std::vector<int>> &output_shapes,
                                               const Target &target) {
  CHECK(attrs.attr_store.count("axes")) << "find no attr of axes";
  std::vector<int> axes = absl::get<std::vector<int>>(attrs.attr_store.at("axes"));

  framework::CINNCompute squeeze_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input arguments of Squeeze compute is empty! Please check.\n";
    CINNValuePack a = args[0];
    CHECK_GE(a.size(), 1U) << "at least 1 input tensors for Squeeze compute\n";
    Expr A = a[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A              = A.as_tensor_ref();
    auto stages                = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    ir::Tensor out = pe::Squeeze(tensor_A, axes, stages, UniqName("Squeeze_out"));
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Squeeze is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule squeeze_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of reshape schedule is empty! Please check.\n";
    CINNValuePack arg_pack = args[0];
    int arg_size           = arg_pack.size();
    poly::StageMap stages  = arg_pack.back();
    Expr out               = arg_pack[0];
    CHECK(out.as_tensor());
    if (target.arch == Target::Arch::NVGPU) {
      pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes[0], target);
    } else if (target.arch == Target::Arch::X86) {
      pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes[0], target);
    }
    *ret = arg_pack;
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(squeeze_compute, squeeze_schedule, "strategy.squeeze.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForSqueeze(const std::vector<std::vector<int>> &inputs_shape,
                                                   const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U) << "The input's shape size should be 1! Please check again.";
  std::vector<int> axes;
  for (auto &iter : attrs) {
    if (iter.first == "axes") {
      axes = absl::get<std::vector<int>>(iter.second);
      break;
    }
  }

  std::vector<int> output_shape;
  int tensor_size = 1;
  if (axes.size()!=0){
    std::vector<int> temp_shape = inputs_shape[0];
    for (auto& a : axes) {
      CHECK(a<temp_shape.size());
      temp_shape[a] = 0;
    }
    for (auto& i : temp_shape) {
      if(i != 0){
        output_shape.push_back(i);
        tensor_size *= i;
      }
    }
  }else{
    for (auto& i : inputs_shape[0]) {
      if(i != 1){
        output_shape.push_back(i);
        tensor_size *= i;
      }
    }
  }

  CHECK(!output_shape.empty()) << "infer_shape for squeeze turns out to be empty. Please check\n";
  int flag_index = -1;
  for (int i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > 0) {
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: " << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == 0) {
      CHECK_LT(i, inputs_shape[0].size())
          << "In op reshape, when attribute shape[i] == 0, shape[i] = input_shape[i]. But now the size of input_shape "
             "<= i, which is incompatible. Please check!";
      output_shape[i] = inputs_shape[0][i];
      CHECK_EQ(tensor_size % output_shape[i], 0)
          << "Incompatible input shape and output shape in op reshape: " << tensor_size << ", " << output_shape[i];
      tensor_size /= output_shape[i];
    } else if (output_shape[i] == -1 && flag_index == -1) {
      flag_index = i;
    } else if (output_shape[i] == -1) {
      LOG(FATAL) << "More than one -1 in output_shape of op reshape.";
    } else {
      LOG(FATAL) << "Unsupported output_shape " << output_shape[i];
    }
  }
  if (flag_index >= 0) output_shape[flag_index] = tensor_size;
  std::vector<std::vector<int>> res{output_shape};
  return res;
}

std::vector<Type> InferDtypeForSqueeze(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<std::vector<std::string>> InferLayoutForSqueeze(const std::vector<framework::shape_t> &input_shapes,
                                                            const std::vector<std::string> &input_layouts,
                                                            const framework::NodeAttr &attrs,
                                                            const Target &target) {
  CHECK_EQ(input_shapes.size(), 1U) << "The input's shape size is not 1! Please check again.";
  CHECK_EQ(input_layouts.size(), 1U) << "The input's layout size is not 1! Please check again.";
  std::vector<std::string> new_input_layouts = input_layouts;
  if (input_shapes[0].size() > 4) {
    // alter input layout back
    new_input_layouts[0] = "NCHW";
    VLOG(3) << "alter input layout from " << input_layouts[0] << " to " << new_input_layouts[0];
  }
  return {new_input_layouts, new_input_layouts};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(squeeze_ops) {
  CINN_REGISTER_OP(squeeze)
      .describe("Squeeze.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForSqueeze)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForSqueeze))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForSqueeze))
      .set_support_level(4);

  return true;
}