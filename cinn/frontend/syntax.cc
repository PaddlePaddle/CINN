#include "cinn/frontend/syntax.h"

#include <memory>
#include <tuple>
#include <utility>

#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {
using hlir::framework::Scope;

void Instruction::PrepareOutputs() {
  auto* op_def = hlir::framework::OpRegistry::Global()->Find(get()->op_type);
  CHECK(op_def) << "No operator called [" << get()->op_type << "]";
  for (int i = 0; i < op_def->num_outputs; i++) {
    get()->outputs.push_back(Variable());
  }
}

Instruction::Instruction(std::string_view op_type, const std::vector<Variable>& inputs, Program* parent)
    : common::Shared<_Instruction_>(common::make_shared<_Instruction_>()) {
  get()->op_type        = op_type;
  get()->parent_program = parent;
  get()->inputs         = inputs;
  PrepareOutputs();
}

Placeholder::operator Variable() const { return var_; }

std::vector<Variable> Program::conv2d(
    const Variable& a,
    const Variable& b,
    const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutputs();
}

Variable Program::batchnorm(const Variable& a,
                            const Variable& b,
                            const std::unordered_map<std::string, hlir::framework::NodeAttr::attr_t>& attr_store) {
  Instruction instr("batchnorm");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutputs()[0];
}

Instruction& Program::operator[](size_t i) {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
}

const Instruction& Program::operator[](size_t i) const {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
}

std::ostream& operator<<(std::ostream& os, const Variable& x) {
  os << "Var(" << x->id << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  std::vector<std::string> outs, ins;
  for (auto& x : instr->inputs) {
    ins.push_back(utils::GetStreamCnt(x));
  }
  for (auto& x : instr->outputs) {
    outs.push_back(utils::GetStreamCnt(x));
  }

  os << utils::Join(outs, ", ") << " = " << instr->op_type << "(" << utils::Join(ins, ", ") << ")";
  return os;
}

class PaddleModelToProgram {
 public:
  explicit PaddleModelToProgram(Scope* scope) : scope_(scope), program_(new Program) {
    CHECK(scope_);

    AddOpMapper_feed();
    AddOpMapper_fetch();
    AddOpMapper_mul();
    AddOpMapper_scale();
  }

  std::unique_ptr<Program> operator()(const std::string& model_dir, bool is_combined) {
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

  // Add an Instruction to a program given a Paddle-format \p op_desc.
  void AddOp(const paddle::cpp::OpDesc& op_desc);

  // @{
  inline void AddOpMapper_feed();
  inline void AddOpMapper_fetch();
  inline void AddOpMapper_scale();
  inline void AddOpMapper_mul();
  // @}

  const std::unordered_map<std::string, Variable>& var_map() const { return var_map_; }

 protected:
  void AddVar(const std::string& name, const Variable& var) {
    CHECK(utils::IsVarNameValid(name));
    CHECK(!var_map_.count(name)) << "Duplicate variable [" << name << "] found";
    var_map_[name] = var;
  }

  Variable GetVar(const std::string& name);

 private:
  std::unordered_map<std::string, std::function<void(const paddle::cpp::OpDesc&)>> op_mappers_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, Variable> var_map_;
  Scope* scope_{};
};

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
    // do nothing
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
      scale              = scale_tensor.mutable_data<float>(common::DefaultHostTarget())[0];
    }

    auto out      = program_->scale(x, scale);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
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
    auto out      = program_->mul(x, y, false, false, x_num_col_dims, y_num_col_dims);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
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
  CHECK(utils::IsVarNameValid(name)) << "Name [" << name << "] is not valid";

  auto it = var_map_.find(name);
  if (it != var_map_.end()) return it->second;

  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = std::get<hlir::framework::Tensor>(*var);
    Variable var;
    var.set_id(name);
    var->shape = tensor.shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    AddVar(name, var);
    return var;
  }

  LOG(FATAL) << "No var called [" << name << "] exists";
  return Variable();
}

std::tuple<std::unique_ptr<Program>, std::unordered_map<std::string, Variable>> LoadPaddleProgram(
    const std::string& model_dir, Scope* scope, bool is_combined) {
  PaddleModelToProgram _(scope);
  return std::make_tuple(_(model_dir, is_combined), _.var_map());
}

void Program::SetInputs(const std::vector<Variable>& xs) {
  CHECK(!xs.empty()) << "At least one input is needed for a program!";
  for (int i = 0; i < xs.size(); i++) {
    CHECK(!xs[i]->shape.empty()) << "Found " << i << "-th input's shape is not set yet";
    CHECK(!xs[i]->type.is_unk()) << "Found " << i << "-th input's type is not set yet";
    inputs_.push_back(xs[i]);
  }
}

void Program::Validate() const {
  CHECK(!inputs_.empty()) << "Inputs of the program is not set yet";
  CHECK(!instrs_.empty()) << "No instruction is added yet";
}

Variable Program::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_add(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu(const Variable& a) {
  Instruction instr("relu", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu6(const Variable& a) {
  Instruction instr("relu6", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::scale(const Variable& a, float ratio) {
  Instruction instr("scale", {a});
  instr.SetAttr("scale", ratio);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::mul(
    const Variable& a, const Variable& b, bool trans_a, bool trans_b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("trans_a", trans_a);
  instr.SetAttr("trans_b", trans_b);
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

}  // namespace frontend
}  // namespace cinn
