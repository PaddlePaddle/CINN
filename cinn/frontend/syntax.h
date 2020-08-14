#pragma once
/**
 * \file This file defines some basic elements of CINN frontend syntax.
 */
#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "cinn/common/context.h"
#include "cinn/common/object.h"
#include "cinn/common/type.h"

namespace cinn {
namespace frontend {

struct Program;
struct Variable;

class Placeholder {
 public:
  Placeholder(const common::Type& type, const std::vector<int>& shape, std::string_view id = "")
      : type_(type), shape_(shape), id_(id.empty() ? common::Context::Global().NewName("placeholder") : id) {}

  const std::vector<int>& shape() const { return shape_; }

  std::string_view id() const { return id_; }

  operator Variable();

  Program* parent_program() { return parent_program_; }

 private:
  common::Type type_;
  std::string id_{};
  std::vector<int> shape_;
  Program* parent_program_{};
};

struct _Variable_ : public common::Object {
  std::string id;
  common::Type type;
  std::vector<int> shape;

  const char* type_info() const override { return __type_info__; }
  static constexpr char* __type_info__ = "cinn_frontend_variable";
};

struct Variable {
  explicit Variable(std::string_view id = "") {
    auto* n = common::make_shared<_Variable_>();
    n->id   = id.empty() ? common::Context::Global().NewName("var") : id;
    data_.Reset(n);
  }

  _Variable_* operator->() { return data_.get(); }
  const _Variable_* operator->() const { return data_.get(); }

 private:
  common::Shared<_Variable_> data_;
};

struct _Instruction_ : public common::Object {
  using attr_t = std::variant<int, float, std::string, std::vector<int>, std::vector<float>, std::vector<std::string>>;

  std::string op_type;
  std::unordered_map<std::string, attr_t> attrs;
  std::vector<Variable> inputs;
  std::vector<Variable> outputs;
  Program* parent_program{};

  const char* type_info() const override { return __type_info__; }

  static constexpr const char* __type_info__ = "cinn_frontend_instruction";
};

struct Instruction : public common::Shared<_Instruction_> {
  explicit Instruction(std::string_view op_type, Program* parent = nullptr);

  void SetInputs(const std::vector<Variable>& vars) { get()->inputs = vars; }
  const std::vector<Variable>& GetOutputs() const { return get()->outputs; }

  template <typename T>
  void SetAttr(const std::string& key, const T& v) {
    get()->attrs[key] = v;
  }

  template <typename T>
  T GetAttr(const std::string& key) {
    auto it = get()->attrs.find(key);
    CHECK(it != get()->attrs.end()) << "No attribute called [" << key << "]";
    return std::get<T>(it->second);
  }

 private:
  // Generate outputs according to op's declaration.
  void PrepareOutputs();
};

struct Program {
  /**
   * Add two variables.
   *
   * @param a The first variable.
   * @param b The second variable.
   * @return The result.
   */
  Variable add(const Variable& a, const Variable& b);

  /**
   * Get \p i-th instruction.
   */
  Instruction& operator[](size_t i);
  /**
   * Get \p i-th instruction.
   */
  const Instruction& operator[](size_t i) const;

  inline size_t size() const { return instrs.size(); }

 private:
  void AddInstruction(const Instruction& other) { instrs.push_back(other); }

  std::vector<Instruction> instrs;
};

std::ostream& operator<<(std::ostream& os, const Variable& x);
std::ostream& operator<<(std::ostream& os, const Instruction& instr);

}  // namespace frontend
}  // namespace cinn
