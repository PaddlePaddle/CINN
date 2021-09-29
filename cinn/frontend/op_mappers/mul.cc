#include "cinn/backends/cuda_util.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void TransposeVar(const std::string& name, const OpMapperContext& ctx) {
  CheckVarNameValid(name);
  auto* var = ctx.scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (ctx.target_.arch == Target::Arch::X86) {
      float* data = tensor->mutable_data<float>(ctx.target_);
      CHECK(tensor->shape().size() == 2) << "The y data's shape size of op [mul] is not equal to 2! Please check.";
      TransposeData(data, tensor->shape().data()[0], tensor->shape().data()[1]);
    } else if (ctx.target_.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
      // To use cublas mul api, there is no need to transpose data.
#ifndef CINN_WITH_CUDNN
      std::vector<float> data(tensor->shape().numel());
      CUDA_CALL(cudaMemcpy(data.data(),
                           reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.target_)),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyDeviceToHost));
      CHECK(tensor->shape().size() == 2) << "The y data's shape size of op [mul] is not equal to 2! Please check.";
      TransposeData(data.data(), tensor->shape().data()[0], tensor->shape().data()[1]);
      CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.target_)),
                           data.data(),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyHostToDevice));
#endif
#else
      LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }

    Variable var;
    var.set_id(name);
    std::vector<int> reverse_shape = tensor->shape().data();
    std::reverse(reverse_shape.begin(), reverse_shape.end());
    tensor->shape().SetData(reverse_shape);
    var->shape = tensor->shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    AddVar(name, var, ctx, true);
  } else {
    LOG(FATAL) << "No var called [" << name << "] exists";
  }
}

void mul_kernel(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  auto x      = utils::GetVar(cinn::utils::TransValidVarName(x_name), ctx);
  TransposeVar(cinn::utils::TransValidVarName(y_name), ctx);
  auto y = utils::GetVar(cinn::utils::TransValidVarName(y_name), ctx);

  auto x_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "x_num_col_dims", 1);
  auto y_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "y_num_col_dims", 1);

  VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
  VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "y shape: " << cinn::utils::Join(y->shape, ",");
  auto out = ctx.builder_->mul(x, y, x_num_col_dims, y_num_col_dims);
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  utils::AddVar(cinn::utils::TransValidVarName(out_name), out, ctx);
  (*ctx.var_model_to_program_map_)[out_name] = out->id;
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(mul) { CINN_REGISTER_OPMAPPER(mul, cinn::frontend::op_mappers::mul_kernel) }
