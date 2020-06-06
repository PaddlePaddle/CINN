#include "cinn/hlir/instruction/primitive/tuple_get.h"

#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct LowerTupleGetLowerImpl : public LowerImplBase {
  explicit LowerTupleGetLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    LowerTupleGet(instr, lower);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, TupleGet, LowerTupleGetLowerImpl)
