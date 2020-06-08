#include "cinn/hlir/instruction/primitive/call.h"

#include "cinn/hlir/instruction/instructions.h"
#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {

struct CallLowerImpl : public LowerImplBase {
  explicit CallLowerImpl(InstrCode code) : LowerImplBase(code) {}

  void Run(const Instruction* instr, Context* context, Scope* scope, ComputationLower* lower) override {
    auto* call = instr->As<CallInstruction>();
    std::vector<Expr> args;
    for (int i = 0; i < instr->operand_count(); i++) {
      lower->LowerInstruction(instr->operand(i));
      auto instr_expr = scope->Lookup(instr->operand(i));
      CHECK(instr_expr.defined());
      args.push_back(instr_expr);
    }

    auto tensors =
        CallImpl(call->computation()->name(), args, call->ret_shapes(), call->ret_tensor_names(), call->ret_types());
    std::vector<Expr> arg_exprs;
    for (auto& tensor : tensors) arg_exprs.emplace_back(tensor);
    lower->AddCallRets(call, arg_exprs);
  }
};

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn

REGISTER_INSTRUCTION_LOWER(base, Call, cinn::hlir::instruction::primitive::CallLowerImpl)
