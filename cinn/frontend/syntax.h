#pragma once
/**
 * \file This file defines some basic elements of CINN frontend syntax.
 */
#include <glog/logging.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/object.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

struct Program;
struct Variable;

struct _Variable_ : public common::Object {
  std::string id;
  common::Type type;
  std::vector<int> shape;

  const char* type_info() const override { return __type_info__; }
  static constexpr char* __type_info__ = "cinn_frontend_variable";
};

/**
 * Variable represents the variable in a computation.
 */
struct Variable : public common::Shared<_Variable_> {
  /**
   * Constructor.
   * @param id_hint The identifier of the variable, if null, a random ID will be assigned.
   */
  explicit Variable(std::string_view id_hint = "") : common::Shared<_Variable_>(common::make_shared<_Variable_>()) {
    if (!id_hint.empty()) CheckVarNameValid(id_hint);
    get()->id = id_hint.empty() ? common::Context::Global().NewName("var") : id_hint;
  }

  void set_id(const std::string& id) { operator->()->id = id; }

  _Variable_* operator->() { return get(); }
  const _Variable_* operator->() const { return get(); }
};

/**
 * Placeholder is the fed slot of a computation.
 */
class Placeholder {
 public:
  /**
   * @param type Type of the fed
   * @param shape Shape of the fed
   * @param id ID of the fed
   */
  Placeholder(const common::Type& type, const std::vector<int>& shape, std::string_view id_hint = "") {
    if (!id_hint.empty()) CheckVarNameValid(std::string(id_hint));
    id_         = id_hint.empty() ? common::Context::Global().NewName("placeholder") : id_hint;
    var_        = Variable(id_);
    var_->shape = shape;
    var_->type  = type;
  }

  const std::vector<int>& shape() const { return var_->shape; }

  Type type() const { return var_->type; }

  std::string_view id() const { return id_; }

  operator Variable() const;

  Program* parent_program() { return parent_program_; }

 private:
  Variable var_;
  std::string id_{};
  Program* parent_program_{};
};

/**
 * Data of a Instruction.
 */
struct _Instruction_ : public common::Object {
  using attr_t = hlir::framework::AttrType;

  std::string op_type;
  std::unordered_map<std::string, attr_t> attrs;
  std::vector<std::pair<std::string, attr_t>> attrs_ordered;
  std::vector<Variable> inputs;
  std::vector<Variable> outputs;
  Program* parent_program{};

  const char* type_info() const override { return __type_info__; }

  std::string debug_string() const;

  static constexpr char* __type_info__ = "cinn_frontend_instruction";
};

/**
 * Instruction is the basic computational unit of a Program, similar to the operator concept in a DNN platform.
 */
struct Instruction : public common::Shared<_Instruction_> {
  explicit Instruction(std::string_view op_type, const std::vector<Variable>& inputs = {}, Program* parent = nullptr);

  /**
   * Set the inputs of the instruction.
   * @param vars The input variables.
   */
  void SetInputs(const std::vector<Variable>& vars) { get()->inputs = vars; }
  const std::vector<Variable>& GetOutputs() const { return get()->outputs; }
  const Variable& GetOutput(size_t offset) const {
    CHECK_LT(offset, get()->outputs.size());
    return GetOutputs()[offset];
  }

  /**
   * Set an attribute of the instruction.
   * @tparam T The type of the attribute value type.
   * @param key The identifier of the attribute.
   * @param v The value of the attribute.
   */
  template <typename T>
  void SetAttr(const std::string& key, const T& v) {
    get()->attrs[key] = v;
  }

  /**
   * Get an attribute of the instruction.
   * @tparam T The data type of the attribute value.
   * @param key The identifier of the attribute.
   * @return The attribute value.
   */
  template <typename T>
  T GetAttrs(const std::string& key) {
    auto it = get()->attrs.find(key);
    CHECK(it != get()->attrs.end()) << "No attribute called [" << key << "]";
    return std::get<T>(it->second);
  }

 private:
  // Generate outputs according to op's declaration.
  void PrepareOutputs();
};

/**
 * Program is a representation of a computation.
 */
struct Program {
  using attr_t = hlir::framework::NodeAttr::attr_t;
  void SetInputs(const std::vector<Variable>& xs);
  /**
   * Add two variables.
   *
   * @param a The first variable.
   * @param b The second variable.
   * @return The result.
   */
  Variable add(const Variable& a, const Variable& b);

  /**
   * Multiply two matrix.
   */
  Variable mul(const Variable& a, const Variable& b, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Multiply two matrix.
   */
  Variable matmul(const Variable& a, const Variable& b, bool trans_a = false, bool trans_b = false, float alpha = 1);

  /**
   * Reshape a tensor.
   */
  Variable reshape2(const Variable& a, const std::vector<int>& shape);

  /**
   * Multiply two matrix and add a bias.
   */
  Variable mulbias(
      const Variable& a, const Variable& b, const Variable& c, int x_num_col_dims = 1, int y_num_col_dims = 1);

  /**
   * Add two tensors element-wise.
   */
  Variable elementwise_add(const Variable& a, const Variable& b, int axis = -1);

  /**
   * Multiply two tensors element-wise.
   */
  Variable elementwise_mul(const Variable& a, const Variable& b, int axis = -1);

  /**
   * Apply Rectified Linear Unit on input Variable.
   * Actually apply: outupt = max(input,0)
   *
   * @param a The first variable.
   * @return The result.
   */
  Variable relu(const Variable& a);
  Variable relu6(const Variable& a);

  /**
   * The convolution2D layer calculates the output based on the input, filter
   * and strides, paddings, dilations, groups parameters.
   *
   * @param a The first variable input.
   * @param b The second variable filter(weights).
   * @param attr_store The params like padding, stride, dilation, etc.
   * @return The result.
   */
  Variable conv2d(const Variable& a, const Variable& b, const std::unordered_map<std::string, attr_t>& attr_store);
  Variable layout_transform(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);
  Variable conv2d_NCHWc(const Variable& a,
                        const Variable& b,
                        const std::unordered_map<std::string, attr_t>& attr_store);
  Variable depthwise_conv2d(const Variable& a,
                            const Variable& b,
                            const std::unordered_map<std::string, attr_t>& attr_store);
  Variable pool2d(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);

  /**
   * The batchnorm layer can be used as a normalizer function
   * for convolution or fully_connected operations.
   *
   * @param a The first variable input.
   * @param b The second variable filter(weights).
   * @param attr_store The params like eplison.
   * @return The result.
   */
  Variable batchnorm(const Variable& a,
                     const Variable& scale,
                     const Variable& bias,
                     const Variable& mean,
                     const Variable& variance,
                     const std::unordered_map<std::string, attr_t>& attr_store);

  Variable scale(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);

  Variable softmax(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);

  Variable sigmoid(const Variable& a);

  Variable slice(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);

  Variable dropout_infer(const Variable& a, const std::unordered_map<std::string, attr_t>& attr_store);

  /**
   * Get \p i-th instruction.
   */
  Instruction& operator[](size_t i);
  /**
   * Get \p i-th instruction.
   */
  const Instruction& operator[](size_t i) const;

  /**
   * Get number of instructions in the program.
   */
  inline size_t size() const { return instrs_.size(); }

  void Validate() const;

  void AppendInstruction(const Instruction& other) { instrs_.push_back(other); }

 private:
  std::vector<Instruction> instrs_;

  std::vector<Variable> inputs_;
};

/**
 * Load a Paddle model and return a frontend program.
 * @param model_dir The directory of the model.
 * @param is_combined Whether the parameters in the Paddle model is combined.
 * @returns program, a map from name to variable and a map from variable name in Paddle model to the corresponding in
 * program
 */
std::tuple<std::unique_ptr<Program>,
           std::unordered_map<std::string, Variable>,
           std::unordered_map<std::string, std::string>>
LoadPaddleProgram(const std::string& model_dir,
                  hlir::framework::Scope* scope,
                  bool is_combined,
                  const common::Target& target = common::DefaultHostTarget());

std::ostream& operator<<(std::ostream& os, const Variable& x);
std::ostream& operator<<(std::ostream& os, const Instruction& instr);
std::ostream& operator<<(std::ostream& os, const Program& program);

}  // namespace frontend
}  // namespace cinn
