#include "cinn/frontend/syntax.h"

#include <iomanip>
#include <memory>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "cinn/frontend/paddle/model_parser.h"
#include "cinn/frontend/paddle_model_to_program.h"
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

Variable Program::conv2d(const Variable& a,
                         const Variable& b,
                         const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::layout_transform(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("layout_transform");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::conv2d_NCHWc(const Variable& a,
                               const Variable& b,
                               const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("conv2d_NCHWc");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::depthwise_conv2d(const Variable& a,
                                   const Variable& b,
                                   const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("depthwise_conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::pool2d(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("pool2d");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::batchnorm(const Variable& a,
                            const Variable& scale,
                            const Variable& bias,
                            const Variable& mean,
                            const Variable& variance,
                            const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("batchnorm");
  instr.SetInputs({a, scale, bias, mean, variance});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::create_const_float(float value, const std::vector<int>& shapes, const std::string& name) {
  Instruction instr("create_const_float");
  Placeholder const_var(Float(32), shapes, name);
  instr.SetInputs({const_var});
  instr.SetAttr("value", value);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::fused_batchnorm_inference(const Variable& a,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& mean,
                                            const Variable& variance,
                                            const std::unordered_map<std::string, attr_t>& attr_store) {
  float epsilon = 0.00001f;
  if (attr_store.find("epsilon") != attr_store.end()) {
    epsilon = std::get<float>(attr_store.at("epsilon"));
  }
  auto eps_var     = create_const_float(epsilon, scale->shape, common::UniqName("epsilon"));
  auto var_add_eps = elementwise_add(eps_var, variance);
  auto rsrqt_var   = primitive_rsqrt(var_add_eps);
  auto new_scale   = elementwise_mul(rsrqt_var, scale);
  auto neg_mean    = primitive_negative(mean);
  auto new_shift   = elementwise_mul(new_scale, neg_mean);
  auto shift_bias  = elementwise_add(new_shift, bias);

  auto temp_out = elementwise_mul(a, new_scale, 1);
  auto bn_out   = elementwise_add(temp_out, shift_bias, 1);

  return bn_out;
}

Variable Program::scale(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("scale", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::softmax(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("softmax", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::sigmoid(const Variable& a) {
  Instruction instr("sigmoid", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::slice(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("slice", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::dropout_infer(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store) {
  Instruction instr("dropout_infer", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
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
  os << instr->debug_string();
  return os;
}

std::tuple<std::unique_ptr<Program>,
           std::unordered_map<std::string, Variable>,
           std::unordered_map<std::string, std::string>>
LoadPaddleProgram(const std::string& model_dir, Scope* scope, bool is_combined, const common::Target& target) {
  LOG(INFO) << "Loading Paddle model from " << model_dir;
  PaddleModelToProgram _(scope, target);
  return std::make_tuple(_(model_dir, is_combined), _.var_map(), _.var_model_to_program_map());
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

#define SYNTAX_UNARY_IMPL(name__)                           \
  Variable Program::primitive_##name__(const Variable& a) { \
    Instruction instr(#name__, {a});                        \
    AppendInstruction(instr);                               \
    return instr.GetOutput(0);                              \
  }

SYNTAX_UNARY_IMPL(exp);
SYNTAX_UNARY_IMPL(erf);
SYNTAX_UNARY_IMPL(sqrt);
SYNTAX_UNARY_IMPL(log);
SYNTAX_UNARY_IMPL(floor);
SYNTAX_UNARY_IMPL(ceil);
SYNTAX_UNARY_IMPL(round);
SYNTAX_UNARY_IMPL(tanh);
SYNTAX_UNARY_IMPL(log2);
SYNTAX_UNARY_IMPL(log10);
SYNTAX_UNARY_IMPL(trunc);
SYNTAX_UNARY_IMPL(cos);
SYNTAX_UNARY_IMPL(sin);
SYNTAX_UNARY_IMPL(cosh);
SYNTAX_UNARY_IMPL(tan);
SYNTAX_UNARY_IMPL(sinh);
SYNTAX_UNARY_IMPL(acos);
SYNTAX_UNARY_IMPL(acosh);
SYNTAX_UNARY_IMPL(asin);
SYNTAX_UNARY_IMPL(asinh);
SYNTAX_UNARY_IMPL(atan);
SYNTAX_UNARY_IMPL(atanh);

SYNTAX_UNARY_IMPL(isnan);
SYNTAX_UNARY_IMPL(isfinite);
SYNTAX_UNARY_IMPL(isinf);
SYNTAX_UNARY_IMPL(bitwise_not);

SYNTAX_UNARY_IMPL(negative);
SYNTAX_UNARY_IMPL(identity);
SYNTAX_UNARY_IMPL(logica_not);
SYNTAX_UNARY_IMPL(sign);
SYNTAX_UNARY_IMPL(abs);
SYNTAX_UNARY_IMPL(rsqrt);

#define SYNTAX_BINARY_IMPL(name__)                                                       \
  Variable Program::primitive_##name__(const Variable& a, const Variable& b, int axis) { \
    Instruction instr(#name__, {a, b});                                                  \
    instr.SetAttr("axis", axis);                                                         \
    AppendInstruction(instr);                                                            \
    return instr.GetOutput(0);                                                           \
  }

SYNTAX_BINARY_IMPL(substract)
SYNTAX_BINARY_IMPL(divide)
SYNTAX_BINARY_IMPL(floor_divide)
SYNTAX_BINARY_IMPL(mod)
SYNTAX_BINARY_IMPL(floor_mod)
SYNTAX_BINARY_IMPL(max)
SYNTAX_BINARY_IMPL(min)
SYNTAX_BINARY_IMPL(power)
SYNTAX_BINARY_IMPL(logical_and)
SYNTAX_BINARY_IMPL(logical_or)
SYNTAX_BINARY_IMPL(logical_xor)
SYNTAX_BINARY_IMPL(greater)
SYNTAX_BINARY_IMPL(less)
SYNTAX_BINARY_IMPL(equal)
SYNTAX_BINARY_IMPL(not_equal)
SYNTAX_BINARY_IMPL(greater_equal)
SYNTAX_BINARY_IMPL(less_equal)

SYNTAX_BINARY_IMPL(bitwise_or)
SYNTAX_BINARY_IMPL(bitwise_xor)
SYNTAX_BINARY_IMPL(bitwise_and)
SYNTAX_BINARY_IMPL(left_shift)
SYNTAX_BINARY_IMPL(right_shift)

Variable Program::elementwise_add(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_mul(const Variable& a, const Variable& b, int axis) {
  Instruction instr("elementwise_mul", {a, b});
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

Variable Program::mul(const Variable& a, const Variable& b, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::mulbias(
    const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims, int y_num_col_dims) {
  Instruction instr("mulbias", {a, b, c});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(1);
}

std::string _Instruction_::debug_string() const {
  struct Visit {
    std::stringstream& s_;
    explicit Visit(std::stringstream& s) : s_(s) {}
    void operator()(int x) { s_ << x; }
    void operator()(float x) { s_ << x; }
    void operator()(bool x) { s_ << (x ? "true" : "false"); }
    void operator()(const std::string& x) { s_ << x; }
    void operator()(const std::vector<int>& x) { s_ << "[" + utils::Join(x, ",") + "]"; }
    void operator()(const std::vector<float>& x) { s_ << "[" + utils::Join(x, ",") + "]"; }
    void operator()(const std::vector<bool>& x) { s_ << "[" + utils::Join(x, ",") + "]"; }
    void operator()(const std::vector<std::string>& x) { s_ << "[" + utils::Join(x, ",") + "]"; }
  };

  std::stringstream ss;
  std::vector<std::string> input_names, output_names;
  std::transform(
      inputs.begin(), inputs.end(), std::back_inserter(input_names), [](const Variable& x) { return x->id; });
  std::transform(
      outputs.begin(), outputs.end(), std::back_inserter(output_names), [](const Variable& x) { return x->id; });

  ss << utils::Join(output_names, ", ");
  ss << " = ";
  ss << op_type;
  ss << "(";
  ss << utils::Join(input_names, ", ");
  if (!attrs.empty()) ss << ", ";

  std::vector<std::string> attr_strs;
  for (auto& attr : attrs) {
    std::stringstream iss;
    iss << attr.first << "=";
    std::visit(Visit{iss}, attr.second);
    attr_strs.push_back(iss.str());
  }
  ss << utils::Join(attr_strs, ", ");
  ss << ")";

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Program& program) {
  os << "Program {\n";
  for (int i = 0; i < program.size(); i++) {
    os << program[i] << "\n";
  }
  os << "}\n";
  return os;
}

}  // namespace frontend
}  // namespace cinn
