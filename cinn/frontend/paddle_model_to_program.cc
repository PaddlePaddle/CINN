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
    CHECK(!op_desc.Input("X").empty());
    auto output_name = op_desc.Input("X").front();
    LOG(INFO) << "detect model output: [" << output_name << "]";
  };
}

void PaddleModelToProgram::AddOpMapper_scale() {
  op_mappers_["scale"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    auto x      = GetVar(utils::TransValidVarName(x_name));
    float scale{};
    if (op_desc.HasAttr("scale")) {  // the old model format
      scale = op_desc.GetAttr<float>("scale");
    } else {  // the newly refactored format
      // load scale tensor
      CHECK(!op_desc.Input("ScaleTensor").empty());
      auto* scale_tensor_var = scope_->FindVar(op_desc.Input("ScaleTensor").front());
      CHECK(scale_tensor_var) << "No scale tensor found in the scope";
      auto& scale_tensor = std::get<hlir::framework::Tensor>(*scale_tensor_var);
      scale              = scale_tensor->mutable_data<float>(common::DefaultHostTarget())[0];
    }
    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    attrs["scale"] = scale;
    auto out       = program_->scale(x, attrs);
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_mul() {
  op_mappers_["mul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Input("Y").empty());
    auto y_name        = op_desc.Input("Y").front();
    auto x             = GetVar(utils::TransValidVarName(x_name));
    auto y             = GetVar(utils::TransValidVarName(y_name));
    int x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
    int y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");
    VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
    VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    VLOG(4) << "y shape: " << utils::Join(y->shape, ",");
    auto out = program_->mul(x, y, x_num_col_dims, y_num_col_dims);
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu() {
  op_mappers_["relu"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();
    auto x        = GetVar(TransValidVarName(x_name));
    auto out      = program_->relu(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_softmax() {
  op_mappers_["softmax"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    if (op_desc.HasAttr("axis")) {
      attrs["axis"] = op_desc.GetAttr<int>("axis");
    } else {
      attrs["axis"] = int(-1);
    }
    auto x   = GetVar(TransValidVarName(x_name));
    auto out = program_->softmax(x, attrs);
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_add() {
  op_mappers_["elementwise_add"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Input("Y").empty());
    auto y_name = op_desc.Input("Y").front();
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();
    int axis      = op_desc.GetAttr<int>("axis");

    auto x   = GetVar(TransValidVarName(x_name));
    auto y   = GetVar(TransValidVarName(y_name));
    auto out = program_->elementwise_add(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu6() {
  op_mappers_["relu6"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("threshold"));
    CHECK_EQ(op_desc.GetAttr<float>("threshold"), 6.0f) << "Threshold of Relu6 is not 6! To be implemented.";
    attrs["threshold"] = op_desc.GetAttr<float>("threshold");

    auto x   = GetVar(TransValidVarName(x_name));
    auto out = program_->relu6(x, attrs);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}
void PaddleModelToProgram::AddOpMapper_depthwise_conv2d() {
  op_mappers_["depthwise_conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("Input").empty());
    auto x_name = op_desc.Input("Input").front();
    CHECK(!op_desc.Input("Filter").empty());
    auto y_name = op_desc.Input("Filter").front();
    CHECK(!op_desc.Output("Output").empty());
    auto out_name = op_desc.Output("Output").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("paddings"));
    attrs["padding"] = op_desc.GetAttr<std::vector<int>>("paddings");
    CHECK(op_desc.HasAttr("strides"));
    attrs["stride"] = op_desc.GetAttr<std::vector<int>>("strides");
    CHECK(op_desc.HasAttr("dilations"));
    attrs["dilation"] = op_desc.GetAttr<std::vector<int>>("dilations");
    CHECK(op_desc.HasAttr("groups"));
    attrs["groups"] = op_desc.GetAttr<int>("groups");
    auto x          = GetVar(TransValidVarName(x_name));
    auto y          = GetVar(TransValidVarName(y_name));
    auto out        = program_->depthwise_conv2d(x, y, attrs);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_conv2d() {
  op_mappers_["conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("Input").empty());
    auto x_name = op_desc.Input("Input").front();
    CHECK(!op_desc.Input("Filter").empty());
    auto y_name = op_desc.Input("Filter").front();
    CHECK(!op_desc.Output("Output").empty());
    auto out_name = op_desc.Output("Output").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("paddings"));
    attrs["padding"] = op_desc.GetAttr<std::vector<int>>("paddings");
    CHECK(op_desc.HasAttr("strides"));
    attrs["stride"] = op_desc.GetAttr<std::vector<int>>("strides");
    CHECK(op_desc.HasAttr("dilations"));
    attrs["dilation"] = op_desc.GetAttr<std::vector<int>>("dilations");
    CHECK(op_desc.HasAttr("groups"));
    attrs["groups"] = op_desc.GetAttr<int>("groups");
    auto x          = GetVar(TransValidVarName(x_name));
    auto y          = GetVar(TransValidVarName(y_name));
    auto out        = program_->conv2d(x, y, attrs);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_pool2d() {
  op_mappers_["pool2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Output("Out").empty());
    auto out_name = op_desc.Output("Out").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("pooling_type"));
    attrs["pool_type"] = op_desc.GetAttr<std::string>("pooling_type");
    CHECK(op_desc.HasAttr("ksize"));
    attrs["kernel_size"] = op_desc.GetAttr<std::vector<int>>("ksize");
    CHECK(op_desc.HasAttr("strides"));
    attrs["stride_size"] = op_desc.GetAttr<std::vector<int>>("strides");
    CHECK(op_desc.HasAttr("paddings"));
    auto padding_size = op_desc.GetAttr<std::vector<int>>("paddings");
    if (padding_size.size() == 2) {
      padding_size.insert(padding_size.begin(), padding_size.front());
      padding_size.push_back(padding_size.back());
    }
    attrs["padding_size"] = padding_size;
    CHECK(op_desc.HasAttr("ceil_mode"));
    attrs["ceil_mode"] = op_desc.GetAttr<bool>("ceil_mode");
    CHECK(op_desc.HasAttr("exclusive"));
    attrs["exclusive"] = op_desc.GetAttr<bool>("exclusive");
    CHECK(op_desc.HasAttr("data_format"));
    attrs["data_format"] = op_desc.GetAttr<std::string>("data_format");

    auto x   = GetVar(TransValidVarName(x_name));
    auto out = program_->pool2d(x, attrs);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_batchnorm() {
  op_mappers_["batch_norm"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK(!op_desc.Input("X").empty());
    auto x_name = op_desc.Input("X").front();
    CHECK(!op_desc.Input("Scale").empty());
    auto scale_name = op_desc.Input("Scale").front();
    CHECK(!op_desc.Input("Bias").empty());
    auto bias_name = op_desc.Input("Bias").front();
    CHECK(!op_desc.Input("Mean").empty());
    auto mean_name = op_desc.Input("Mean").front();
    CHECK(!op_desc.Input("Variance").empty());
    auto variance_name = op_desc.Input("Variance").front();
    CHECK(!op_desc.Output("Y").empty());
    auto out_name = op_desc.Output("Y").front();

    std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("epsilon"));
    attrs["epsilon"] = op_desc.GetAttr<float>("epsilon");
    auto x           = GetVar(TransValidVarName(x_name));
    auto scale       = GetVar(TransValidVarName(scale_name));
    auto bias        = GetVar(TransValidVarName(bias_name));
    auto mean        = GetVar(TransValidVarName(mean_name));
    auto variance    = GetVar(TransValidVarName(variance_name));
    auto out         = program_->batchnorm(x, scale, bias, mean, variance, attrs);

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
