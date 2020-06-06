#include "cinn/hlir/instruction/primitive/tuple.h"

#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct TupleLowerImpl : public LowerImplBase {
  explicit TupleLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    // Do nothing
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Tuple, TupleLowerImpl)
