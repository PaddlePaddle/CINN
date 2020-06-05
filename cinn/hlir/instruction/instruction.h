#pragma once
#include <glog/logging.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/type.h"
#include "cinn/hlir/instruction/instr_code.h"
#include "cinn/hlir/instruction/shape.h"

namespace cinn {
namespace hlir {
namespace instruction {
using type_t = cinn::common::Type;
using cinn::common::Bool;
using cinn::common::Float;
using cinn::common::Int;
using cinn::common::UInt;
using cinn::common::Void;

struct InstructionKind {
  using item_t = unsigned int;
  enum class Kind : item_t {
    Elementwise = 1,  //!< Elementwise operations.
  };

  bool is_elementwise() const { return TellFlag(KindAsInt(Kind::Elementwise)); }

  InstructionKind& set_elementwise(bool x);

  static inline item_t KindAsInt(Kind kind) { return static_cast<item_t>(kind); }

 private:
  void SetFlag(unsigned int flag, bool x);

  inline bool TellFlag(item_t flag) const { return KindAsInt(kind_) & flag; }

 private:
  Kind kind_;
};

enum class CompareDirection { LT = 0, LE, GT, GE, EQ };

class Computation;

struct ParameterConfig {
  type_t type;
};

struct ConstantConfig {
  type_t type;
};

/**
 * The Instruction is a higher level of compiler's IR, it is a atomic unit lower than operators in DNN platforms.
 * Instructions lives on the instruction layer, the upper operator layer lowers to this layer. It does not have basic
 * blocks.
 */
class Instruction {
 public:
  //! Creates a parameter-retrieving instruction.
  static std::unique_ptr<Instruction> CreateParameter(int param_offset,
                                                      const Shape& shape,
                                                      const std::string& name,
                                                      const ParameterConfig& config);

  //! Create an unary instruction.
  static std::unique_ptr<Instruction> CreateUnary(InstrCode instr_code,
                                                  Instruction* arg0,
                                                  const Shape& shape = Shape());

  //! Create an binary instruction.
  static std::unique_ptr<Instruction> CreateBinary(InstrCode instr_code,
                                                   Instruction* arg0,
                                                   Instruction* arg1,
                                                   const Shape& shape = Shape());

  static std::unique_ptr<Instruction> CreateCompare(Instruction* arg0,
                                                    Instruction* arg1,
                                                    CompareDirection dire,
                                                    const Shape& shape = Shape());

  static std::unique_ptr<Instruction> CreateDot(Instruction* arg0, Instruction* arg1, const Shape& shape = Shape());

  static std::unique_ptr<Instruction> CreateReduce(Instruction* operand,
                                                   Instruction* init_value,
                                                   const std::vector<int>& reduce_dimensions,
                                                   Computation* reduce_computation,
                                                   const Shape& shape = Shape());

  static std::unique_ptr<Instruction> CreateBroadcast(const Shape& shape,
                                                      Instruction* arg0,
                                                      const std::vector<int>& dimensions);

  static std::unique_ptr<Instruction> CreateTranspose(const Shape& shape,
                                                      Instruction* arg0,
                                                      const std::vector<int>& dimensions);

  // Call with single return tensor.
  static std::unique_ptr<Instruction> CreateCall(const std::vector<Instruction*>& args,
                                                 const std::string& ret_name,
                                                 const Shape& shape,
                                                 const cinn::common::Type& type,
                                                 const Computation* computation);
  /**
   * Get a call with multiple return values.
   * @param args The call arguments.
   * @param ret_names The return names.
   * @param ret_types The return types.
   * @param ret_shapes The return shapes.
   * @param computation The computation to call.
   * @return The call instruction.
   */
  static std::unique_ptr<Instruction> CreateCall(const std::vector<Instruction*>& args,
                                                 const std::vector<Shape>& ret_names,
                                                 const std::vector<cinn::common::Type>& ret_types,
                                                 const std::vector<Shape>& ret_shapes,
                                                 Computation* computation);

  // Get a call instruction.
  static std::unique_ptr<Instruction> CreateCustomCall(const Shape& shape,
                                                       const std::vector<Instruction*>& args,
                                                       const std::string& target,
                                                       const std::string& tag);

  static std::unique_ptr<Instruction> CreateTupleGet(Instruction* tuple, int offset);

  static std::unique_ptr<Instruction> CreateTuple(Instruction* call);
  static std::unique_ptr<Instruction> CreateTuple(const std::vector<Instruction*>& items);

  static std::unique_ptr<Instruction> CreateNary(const Shape& shape,
                                                 const std::vector<Instruction*>& args,
                                                 InstrCode instr_code);

  static std::unique_ptr<Instruction> CreateConstant(const Shape& shape,
                                                     const std::vector<char>& buf,
                                                     const ConstantConfig& config);

  static std::unique_ptr<Instruction> CreateConv(
      Instruction* I, Instruction* W, int pad_h, int pad_w, int stride_h, int stridd_w);

  template <typename T>
  T* As() {
    static_assert(std::is_base_of<Instruction, T>());
    return static_cast<T*>(this);
  }
  template <typename T>
  const T* As() const {
    static_assert(std::is_base_of<Instruction, T>());
    return static_cast<const T*>(this);
  }

  //! Add an operand.
  void AppendOperand(Instruction* operand);

  //! Get the i-th operand.
  const Instruction* operand(int i) const;

  //! Get the number of operands of the instruction.
  inline size_t operand_count() const { return operands_.size(); }

  //! Adds a control dependency form this instruction to the given one.
  void AddControlDependencyTo(Instruction* instruction) { inlinks_.insert(instruction); }
  void RemoveControlDependency(Instruction* instruction) { inlinks_.erase(instruction); }

  virtual std::string id() const { return std::to_string(id_); }
  virtual std::string programable_id() const;

  const void* belonged_computation_builder() const { return belonged_computation_builder_; }
  void* belonged_computation_builder() { return belonged_computation_builder_; }
  void set_belonged_computation_builder(void* x) { belonged_computation_builder_ = x; }

  virtual std::string to_debug_string() const;

  inline const Shape& shape() const { return shape_; }
  inline const type_t& type() const { return type_; }
  inline std::string comment() const { return comment_.value_or(""); }
  inline InstrCode instr_code() const { return instr_code_; }

  inline void set_comment(const std::string& comment) { comment_ = comment; }
  inline void set_type(const type_t& type);

  bool inlined() const { return inlined_; }
  void set_inlined(bool x = true) { inlined_ = x; }

  //! Add usage relation.
  void AddUser(Instruction* user);
  //! Remove usage relation.
  void RemoveUser(Instruction* user);

  void set_lower_kind(const std::string& x) { lower_kind_ = x; }
  const std::string& kind() const { return lower_kind_; }

 protected:
  Instruction(InstrCode code, const Shape& shape) : instr_code_(code), shape_(shape) {}

  void set_id(int id) { id_ = id; }

  friend class Computation;

 protected:
  int id_{-1};

 protected:
  InstrCode instr_code_;
  Shape shape_;
  std::vector<Instruction*> operands_;
  std::set<Instruction*> inlinks_;
  std::set<Instruction*> outlinks_;
  std::vector<const Computation*> called_computations_;
  std::optional<std::string> comment_;
  bool inlined_{false};
  type_t type_{Void()};
  void* belonged_computation_builder_{};
  //! Control which kind of the LowerImpl this instruction will use, this field can set by some analysis pass.
  std::string lower_kind_{"base"};
};

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
