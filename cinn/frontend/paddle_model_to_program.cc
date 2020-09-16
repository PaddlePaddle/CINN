#include "cinn/frontend/paddle_model_to_program.h"

#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/frontend/paddle/pb/program_desc.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace frontend {
using utils::Join;
using utils::TransValidVarName;

void PaddleModelToProgram::AddOpMapper_feed() {
  op_mappers_["feed"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto outs = op_desc.Output("Out");
    CHECK_EQ(outs.size(), 1UL);
    VLOG(2) << "Model get feed [" << outs[0] << "]";
    Placeholder input(Float(32), {}, outs[0]);
    AddVar(outs[0], input);
  };
}

void PaddleModelToProgram::AddOpMapper_fetch() {
  op_mappers_["fetch"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto output_name = op_desc.Input("X").front();
    LOG(INFO) << "detect model output: [" << output_name << "]";
  };
}

void PaddleModelToProgram::AddOpMapper_scale() {
  op_mappers_["scale"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto x_name = op_desc.Input("X").front();
    auto x      = GetVar(utils::TransValidVarName(x_name));
    float scale{};
    if (op_desc.HasAttr("scale")) {  // the old model format
      scale = op_desc.GetAttr<float>("scale");
    } else {  // the newly refactored format
      // load scale tensor
      auto* scale_tensor_var = scope_->FindVar(op_desc.Input("ScaleTensor").front());
      CHECK(scale_tensor_var) << "No scale tensor found in the scope";
      auto& scale_tensor = std::get<hlir::framework::Tensor>(*scale_tensor_var);
      scale              = scale_tensor->mutable_data<float>(common::DefaultHostTarget())[0];
    }
    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    attrs["scale"] = scale;
    auto out       = program_->scale(x, attrs);
    auto out_name  = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_mul() {
  op_mappers_["mul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto x_name        = op_desc.Input("X").front();
    auto y_name        = op_desc.Input("Y").front();
    auto x             = GetVar(utils::TransValidVarName(x_name));
    auto y             = GetVar(utils::TransValidVarName(y_name));
    int x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
    int y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");
    VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
    VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    VLOG(4) << "y shape: " << utils::Join(y->shape, ",");
    auto out      = program_->mul(x, y, false, false, x_num_col_dims, y_num_col_dims);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu() {
  op_mappers_["relu"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto x_name   = op_desc.Input("X").front();
    auto out_name = op_desc.Output("Out").front();
    auto x        = GetVar(TransValidVarName(x_name));
    auto out      = program_->relu(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_add() {
  op_mappers_["elementwise_add"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto x_name   = op_desc.Input("X").front();
    auto y_name   = op_desc.Input("Y").front();
    auto out_name = op_desc.Output("Out").front();
    int axis      = op_desc.GetAttr<int>("axis");

    auto x   = GetVar(TransValidVarName(x_name));
    auto y   = GetVar(TransValidVarName(y_name));
    auto out = program_->elementwise_add(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOp(const paddle::cpp::OpDesc& op_desc) {
  const auto& op_type = op_desc.Type();
  auto it             = op_mappers_.find(op_type);
  if (it != op_mappers_.end()) {
    it->second(op_desc);
    return;
  }
  // feed op's output is a input of the model
  LOG(FATAL) << "Not supported op [" << op_desc.Type() << "] found";
}

Variable PaddleModelToProgram::GetVar(const std::string& name) {
  CheckVarNameValid(name);

  auto it = var_map_.find(name);
  if (it != var_map_.end()) return it->second;

  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = std::get<hlir::framework::Tensor>(*var);
    Variable var;
    var.set_id(name);
    var->shape = tensor->shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    AddVar(name, var);
    return var;
  }

  LOG(FATAL) << "No var called [" << name << "] exists";
  return Variable();
}

std::unique_ptr<Program> PaddleModelToProgram::operator()(const std::string& model_dir, bool is_combined) {
  paddle::cpp::ProgramDesc program_desc;
  paddle::LoadModelPb(model_dir, "__model__", "", scope_, &program_desc, is_combined);
  CHECK_EQ(program_desc.BlocksSize(), 1) << "CINN can only support the model with a single block";
  auto* block_desc = program_desc.GetBlock<paddle::cpp::BlockDesc>(0);

  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    AddOp(*op_desc);
  }
  return std::move(program_);
}

void PaddleModelToProgram::AddVar(const std::string& name, const Variable& var) {
  CheckVarNameValid(name);
  CHECK(!var_map_.count(name)) << "Duplicate variable [" << name << "] found";
  var_map_[name] = var;
}

}  // namespace frontend
}  // namespace cinn
