#pragma once
#include <glog/logging.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "cinn/ir/ir.h"
#include "hlir/instruction/instr_code.h"
#include "hlir/instruction/shape.h"

namespace hlir {
namespace instruction {

struct InstructionKind {
  using item_t = unsigned int;
  enum class Kind : item_t {
    Elementwise = 1,  //!< Elementwise operations.
  };

  bool is_elementwise() const { return TellFlag(KindAsInt(Kind::Elementwise)); }

  InstructionKind& set_elementwise(bool x) {
    SetFlag(KindAsInt(Kind::Elementwise), x);
    return *this;
  }

  static inline item_t KindAsInt(Kind kind) { return static_cast<item_t>(kind); }

 private:
  void SetFlag(unsigned int flag, bool x);

  inline bool TellFlag(item_t flag) const { return KindAsInt(kind_) & flag; }

 private:
  Kind kind_;
};

enum class CompareDirection { LT = 0, LE, GT, GE, EQ };

class Computation;

/**
 * The Instruction is a higher level of compiler's IR, it is a atomic unit lower than operators in DNN platforms.
 * Instructions lives on the instruction layer, the upper operator layer lowers to this layer. It does not have basic
 * blocks.
 */
class Instruction {
 public:
  //! Creates a parameter-retrieving instruction.
  static std::unique_ptr<Instruction> CreateParameter(const Shape& shape, const std::string& name);

  //! Create an unary instruction.
  static std::unique_ptr<Instruction> CreateUnary(const Shape& shape, InstrCode instr_code, Instruction* arg0);

  //! Create an binary instruction.
  static std::unique_ptr<Instruction> CreateBinary(const Shape& shape,
                                                   InstrCode instr_code,
                                                   Instruction* arg0,
                                                   Instruction* arg1);

  static std::unique_ptr<Instruction> CreateCompare(const Shape& shape,
                                                    Instruction* arg0,
                                                    Instruction* arg1,
                                                    CompareDirection dire);

  static std::unique_ptr<Instruction> CreateDot(const Shape& shape, Instruction* arg0, Instruction* arg1);

  static std::unique_ptr<Instruction> CreateReduce(const Shape& shape,
                                                   Instruction* operand,
                                                   Instruction* init_value,
                                                   const std::vector<int>& reduce_dimensions,
                                                   Computation* reduce_computation);

  static std::unique_ptr<Instruction> CreateBroadcast(const Shape& shape,
                                                      Instruction* arg0,
                                                      const std::vector<int>& dimensions);

  static std::unique_ptr<Instruction> CreateTranspose(const Shape& shape,
                                                      Instruction* arg0,
                                                      const std::vector<int>& dimensions);

  static std::unique_ptr<Instruction> CreateCall(const Shape& shape,
                                                 const std::vector<Instruction*>& args,
                                                 Computation* computation);

  static std::unique_ptr<Instruction> CreateNary(const Shape& shape,
                                                 const std::vector<Instruction*>& args,
                                                 InstrCode instr_code);

  //! Add an operand.
  void AppendOperand(Instruction* operand);

  //! Get the i-th operand.
  const Instruction* operand(int i) const;

  //! Get the number of operands of the instruction.
  size_t operand_count() const { return operands_.size(); }

  //! Adds a control dependency form this instruction to the given one.
  void AddControlDependencyTo(Instruction* instruction) { inlinks_.insert(instruction); }
  void RemoveControlDependency(Instruction* instruction) { inlinks_.erase(instruction); }

 protected:
  Instruction(InstrCode code, const Shape& shape) : instr_code_(code), shape_(shape) {}

 private:
  InstrCode instr_code_;
  Shape shape_;
  std::vector<Instruction*> operands_;
  std::set<Instruction*> inlinks_;
  std::set<Instruction*> outlinks_;
  std::vector<Computation*> called_computations_;
};

}  // namespace instruction
}  // namespace hlir
