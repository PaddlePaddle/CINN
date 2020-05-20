#include "cinn/backends/llvm/codegen_llvm.h"

#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <type_traits>

#include "cinn/backends/extern_func_emitter.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/common/type.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

using BinaryInstruction = llvm::Instruction::BinaryOps;

namespace {

template <typename T>
auto NodeToExpr(const T *node) {
  std::ostringstream oss;
  oss << "\033[32m";
  oss << ir::Expr(const_cast<T *>(node));
  oss << "\033[0m";
  return oss.str();
}

bool is_integral_type(common::Type t) { return t.is_int() || t.is_uint(); }

bool is_floating_type(common::Type t) { return t.is_float(); }

llvm::Value *EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value *lhs,
                            llvm::Value *rhs,
                            llvm::IRBuilder<> *b) {
  llvm::Value *comparison_result{nullptr};
  if (lhs->getType()->isIntegerTy()) {
    comparison_result = b->CreateICmp(predicate, lhs, rhs);
  } else {
    comparison_result = b->CreateFCmp(predicate, lhs, rhs);
  }

  return b->CreateZExt(comparison_result, b->getInt8Ty());
  // return b->CreateZExt(comparison_result,
  // llvm::Type::getInt8Ty(m->getContext()));
}

#define __IR_EMITTER_NOT_IMPLEMENTED(__op)                                  \
  LOG(INFO) << "Error: file[" << __FILE__ << "], line[" << __LINE__ << "]"; \
  throw std::logic_error("Not implemented error!")

}  // namespace

CodeGenLLVM::CodeGenLLVM(llvm::Module *m,
                         llvm::IRBuilder<> *b,
                         std::shared_ptr<std::unordered_map<std::string, llvm::Value *>> vars)
    : m_(m), b_(b), named_vars_(vars) {
  if (!named_vars_.get()) {
    named_vars_ = std::make_shared<std::unordered_map<std::string, llvm::Value *>>();
  }
}

CodeGenLLVM::~CodeGenLLVM() {}

llvm::Value *CodeGenLLVM::EmitBinaryOp(
    llvm::Value *lhs, llvm::Value *rhs, char opcode, bool is_integral, bool is_signed) {
  llvm::Instruction::BinaryOps ops;
  switch (opcode) {
    case '+':
      ops = is_integral ? llvm::Instruction::BinaryOps::Add : llvm::Instruction::BinaryOps::FAdd;
      break;
    case '-':
      ops = is_integral ? llvm::Instruction::BinaryOps::Sub : llvm::Instruction::BinaryOps::FSub;
      break;
    case '*':
      ops = is_integral ? llvm::Instruction::BinaryOps::Mul : llvm::Instruction::BinaryOps::FMul;
      break;
    case '/':
      ops = is_integral ? (is_signed ? llvm::Instruction::BinaryOps::SDiv : llvm::Instruction::BinaryOps::UDiv)
                        : llvm::Instruction::BinaryOps::FDiv;
      break;
    case '%':
      ops = is_integral ? (is_signed ? llvm::Instruction::BinaryOps::SRem : llvm::Instruction::BinaryOps::URem)
                        : llvm::Instruction::BinaryOps::FRem;
      break;
    default:
      return nullptr;
  }
  return BinOp(ops, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::IntImm *op) {
  auto *type = b_->getIntNTy(op->type().bits());
  return llvm::ConstantInt::get(type, op->value, true);
}

llvm::Value *CodeGenLLVM::Visit(const ir::UIntImm *op) {
  auto *type = b_->getIntNTy(op->type().bits());
  return llvm::ConstantInt::get(type, op->value, false);
}

llvm::Value *CodeGenLLVM::Visit(const ir::FloatImm *op) { return llvm::ConstantFP::get(b_->getFloatTy(), op->value); }

llvm::Value *CodeGenLLVM::LLVMGenGlobalStringVar(const std::string &data) { return b_->CreateGlobalStringPtr(data); }

llvm::Value *CodeGenLLVM::Visit(const ir::StringImm *op) { return LLVMGenGlobalStringVar(op->value); }

llvm::Value *CodeGenLLVM::Visit(const ir::Add *op) {
  return EmitBinaryOp(Visit(&op->a()), Visit(&op->b()), '+', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Sub *op) {
  return EmitBinaryOp(Visit(&op->a()), Visit(&op->b()), '-', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Mul *op) {
  return EmitBinaryOp(Visit(&op->a()), Visit(&op->b()), '*', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Div *op) {
  return EmitBinaryOp(Visit(&op->a()), Visit(&op->b()), '/', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Mod *op) {
  return EmitBinaryOp(Visit(&op->a()), Visit(&op->b()), '%', is_integral_type(op->type()));
}

#define __IR_EMITTER_DEFINE_CMP_VISITOR(__sop, __uop, __fop) \
  auto *lhs = Visit(&op->a());                               \
  auto *rhs = Visit(&op->b());                               \
  CHECK(op->a().type() == op->b().type());                   \
  llvm::CmpInst::Predicate predicate;                        \
  if (op->a().type().is_int()) {                             \
    predicate = llvm::CmpInst::ICMP_##__sop;                 \
  } else if (op->a().type().is_uint()) {                     \
    predicate = llvm::CmpInst::ICMP_##__uop;                 \
  } else /*float*/ {                                         \
    predicate = llvm::CmpInst::FCMP_##__fop;                 \
  }                                                          \
  return EmitComparison(predicate, lhs, rhs, b_)

llvm::Value *CodeGenLLVM::Visit(const ir::EQ *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(EQ, EQ, OEQ); }

llvm::Value *CodeGenLLVM::Visit(const ir::NE *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(NE, NE, ONE); }

llvm::Value *CodeGenLLVM::Visit(const ir::LT *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(SLT, ULT, OLT); }

llvm::Value *CodeGenLLVM::Visit(const ir::LE *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(SLE, ULE, OLE); }

llvm::Value *CodeGenLLVM::Visit(const ir::GT *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(SGT, UGT, OGT); }

llvm::Value *CodeGenLLVM::Visit(const ir::GE *op) { __IR_EMITTER_DEFINE_CMP_VISITOR(SGE, UGE, OGE); }

#undef __IR_EMITTER_DEFINE_CMP_VISITOR

llvm::Value *CodeGenLLVM::Visit(const ir::And *op) { return And(Visit(&op->a()), Visit(&op->b())); }

llvm::Value *CodeGenLLVM::Visit(const ir::Or *op) { return Or(Visit(&op->a()), Visit(&op->b())); }

llvm::Value *CodeGenLLVM::Visit(const ir::Min *op) {
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());

  llvm::Value *p{nullptr};
  if (op->type().is_int()) {
    p = ICmpSLT(lhs, rhs);
  } else if (op->type().is_uint()) {
    p = ICmpULT(lhs, rhs);
  } else /*float*/ {
    p = FCmpOLT(lhs, rhs);
  }

  return Select(p, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Max *op) {
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());

  llvm::Value *p = nullptr;
  if (op->type().is_int()) {
    p = ICmpSGT(lhs, rhs);
  } else if (op->type().is_uint()) {
    p = ICmpUGT(lhs, rhs);
  } else /*float*/ {
    p = FCmpOGT(lhs, rhs);
  }

  return Select(p, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Minus *op) {
  auto *v = Visit(&op->v());
  return (op->type().is_int() || op->type().is_uint()) ? Neg(v) : FNeg(v);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Not *op) { return Not(Visit(&op->v())); }

llvm::Value *CodeGenLLVM::Visit(const ir::Cast *op) {
  auto from = op->v().type();
  auto to   = op->type();

  llvm::Type *source = CinnTypeToIrType(from, m_);
  llvm::Type *target = CinnTypeToIrType(to, m_);
  CHECK(source) << "source ir type is null";
  CHECK(target) << "target ir type is null";

  llvm::Value *value = Visit(&op->v());
  CHECK(value) << "value is null";

  // pod_value_t cast to a value.
  if (op->v().type().is_customized_type() &&
      op->v().type().customized_type() == common::customized_type::kpod_value_t) {  // pod_value_t operator
    llvm::Function *callee{};
    if (op->type().is_int(32)) {
      callee = m_->getFunction("cinn_pod_value_to_int32");
    } else if (op->type().is_int(64)) {
      callee = m_->getFunction("cinn_pod_value_to_int64");
    } else if (op->type().is_float(32)) {
      callee = m_->getFunction("cinn_pod_value_to_float");
    } else if (op->type().is_float(64)) {
      callee = m_->getFunction("cinn_pod_value_to_double");
    } else if (op->type() == type_of<void *>()) {
      callee = m_->getFunction("cinn_pod_value_to_void_p");
    } else if (op->type() == type_of<cinn_buffer_t *>() || op->type() == type_of<const cinn_buffer_t *>()) {
      callee = m_->getFunction("cinn_pod_value_to_buffer_p");
    } else {
      LOG(ERROR) << "can't cast cinn_pod_value_t to " << op->type();
      NOT_IMPLEMENTED
    }
    CHECK(callee);

    return Call(callee, std::vector<llvm::Value *>({value}), "pod_value_cast");
  }

  do {
    if (value->getType() == target) break;

    if (to.is_cpp_handle() || to.is_cpp_handle_handle()) {
      value = BitCast(value, target);
      break;
    }

    if (to.is_bool()) {
      if (from.is_float()) {
        llvm::Constant *zero = llvm::ConstantFP::get(source, 0.);
        value                = FCmpONE(value, zero);
      } else {
        llvm::Constant *zero = llvm::ConstantInt::get(source, 0);
        value                = ICmpNE(value, zero);
      }
      break;
    }

    if (from.is_float() == false && to.is_float() == false) {
      value = IntCast(value, target, from.is_int());
      break;
    }

    if (from.is_float() && to.is_int()) {
      value = FPToSI(value, target);
      break;
    }

    if (from.is_float() && to.is_uint()) {
      value = FPToUI(value, target);
      if (to.bits() < 8) {
        value = IntCast(value, target, false);
      }
      break;
    }

    if (from.is_int() && to.is_float()) {
      value = SIToFP(value, target);
      break;
    }

    if (from.is_uint() && to.is_float()) {
      value = UIToFP(value, target);
      break;
    }

    CHECK(from.is_float() && to.is_float());
    value = FPCast(value, target);
  } while (false);

  return value;
}

llvm::Value *CodeGenLLVM::Visit(const ir::For *op) {
  llvm::BasicBlock *preheader_bb = b_->GetInsertBlock();
  llvm::BasicBlock *exit_bb      = nullptr;

  llvm::BasicBlock::iterator insert_point = b_->GetInsertPoint();

  if (insert_point == preheader_bb->end()) {
    CHECK(!preheader_bb->getTerminator());
    exit_bb = llvm::BasicBlock::Create(b_->getContext(), "loop_exit", b_->GetInsertBlock()->getParent(), nullptr);
  } else {
    CHECK(preheader_bb->getTerminator());
    exit_bb = preheader_bb->splitBasicBlock(insert_point, "loop_exit");
    preheader_bb->getTerminator()->eraseFromParent();
  }

  llvm::BasicBlock *header_bb =
      llvm::BasicBlock::Create(b_->getContext(), "loop_header", b_->GetInsertBlock()->getParent(), nullptr);
  llvm::BasicBlock *body_bb =
      llvm::BasicBlock::Create(b_->getContext(), "loop_body", b_->GetInsertBlock()->getParent(), nullptr);

  llvm::Function *func = preheader_bb->getParent();
  b_->SetInsertPoint(&func->getEntryBlock(), func->getEntryBlock().getFirstInsertionPt());

  llvm::Value *old_var = GetVar(op->loop_var->name);
  // loop iterator
  llvm::AllocaInst *loop_var = Alloca(b_->getInt32Ty(), nullptr, op->loop_var->name);
  SetVar(op->loop_var->name, loop_var);

  b_->SetInsertPoint(preheader_bb);
  llvm::Value *start_index = Visit(&op->min);
  llvm::Value *end_index   = Visit(&op->extent);
  Store(start_index, loop_var);
  CHECK(!preheader_bb->getTerminator());
  Br(header_bb);

  // loop_header
  b_->SetInsertPoint(header_bb);
  llvm::Value *indvar    = Load(loop_var, "indvar");
  llvm::Value *exit_cond = ICmpSGE(indvar, end_index);
  CondBr(/*Cond=*/exit_cond,
         /*True=*/exit_bb,
         /*False=*/body_bb);

  // loop_body
  b_->SetInsertPoint(body_bb);
  // TODO(fc500110) support step > 1
  llvm::Value *step = llvm::ConstantInt::get(b_->getInt32Ty(), 1);

  Visit(&op->body);
  llvm::Value *indvar_inc = Add(indvar,
                                step,
                                "indvar.inc",
                                /*HasNUW=*/true,
                                /*HasNSW=*/true);
  Store(indvar_inc, loop_var);
  Br(header_bb);

  if (old_var) {
    SetVar(op->loop_var->name, old_var);
  } else {
    named_vars_->erase(op->loop_var->name);
  }

  b_->SetInsertPoint(exit_bb);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::PolyFor *op) {
  NOT_IMPLEMENTED
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Select *op) {
  return Select(Visit(&op->condition), Visit(&op->true_value), Visit(&op->false_value));
}

llvm::Value *CodeGenLLVM::Visit(const ir::IfThenElse *op) {
  bool emit_else = op->false_case.defined();

  llvm::BasicBlock *if_block   = b_->GetInsertBlock();
  llvm::BasicBlock *true_block = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"if-true",
      /*Parent=*/b_->GetInsertBlock()->getParent(),
      /*InsertBefore=*/nullptr);
  llvm::BasicBlock *false_block = nullptr;
  if (emit_else) {
    false_block = llvm::BasicBlock::Create(b_->getContext(), "if-false", b_->GetInsertBlock()->getParent(), nullptr);
  }

  llvm::BasicBlock *after_block = nullptr;
  if (if_block->getTerminator() == nullptr) {
    b_->SetInsertPoint(if_block);
    after_block = llvm::BasicBlock::Create(b_->getContext(), "if-after", b_->GetInsertBlock()->getParent(), nullptr);
    Br(after_block);
  } else {
    after_block = if_block->splitBasicBlock(b_->GetInsertPoint(), "if-after");
  }

  b_->SetInsertPoint(if_block);
  Visit(&op->true_case);
  CondBr(Visit(&op->condition), true_block, emit_else ? false_block : after_block);

  b_->SetInsertPoint(true_block);
  Br(after_block);

  if (emit_else) {
    b_->SetInsertPoint(false_block);
    Visit(&op->false_case);
    Br(after_block);
  }

  b_->SetInsertPoint(after_block, after_block->getFirstInsertionPt());

  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Block *op) {
  llvm::Value *ret = nullptr;

  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(b_->getContext(), "block", b_->GetInsertBlock()->getParent(), nullptr);

  Br(block);
  b_->SetInsertPoint(block);

  for (const auto &expr : op->stmts) {
    ret = Visit(&expr);
  }

  return ret;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Call *op) {
  if (op->name == runtime::buffer_create) {
  } else if (op->name == runtime::get_address_repr) {
    return EmitCall_get_address(op);
  } else if (op->name == runtime::debug_log_repr) {
    return EmitCall_debug_info(op);
  } else if (op->is_extern_call()) {
    auto emitter_id = ExternFuncID{backend_llvm_host, op->name.c_str()};
    auto *emitter   = ExternFunctionEmitterRegistry::Global().Lookup(emitter_id);
    CHECK(emitter) << "No extern function emitter called " << emitter_id;
    emitter->BindCodeGen(this);
    emitter->Emit(op);
    return extern_func_emit_res_;
  }

  llvm::Function *callee = m_->getFunction(op->name);
  CHECK(callee) << "Unknown function referenced. [" << op->name << "]";

  std::vector<llvm::Value *> args;
  for (const auto &e : op->read_args) {
    args.push_back(Visit(&e));
  }
  for (const auto &e : op->write_args) {
    args.push_back(Visit(&e));
  }

  return Call(callee, std::move(args), "calltmp");
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Module_ *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::_Var_ *op) {
  llvm::Value *value = GetVar(op->name, false);
  CHECK(value) << "ir::_Var_[" << op->name << "]: value is null";
  // TODO(fc500110) hard coding
  if (op->name == "_args") return value;
  if (value->getType()->isPointerTy()) {
    return Load(value);
  }
  return value;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Load *op) {
  llvm::Value *array{nullptr};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array = GetVar(var_op->name);
  } else {
    array = Visit(&op->tensor);
  }
  CHECK(array) << "fail to Visit Load node: " << Expr(const_cast<ir::Load *>(op));

  ir::Expr index = op->index();
  std::vector<llvm::Value *> indices;
  indices.push_back(Visit(&index));

  auto res = Load(GEP(array, std::move(indices)));
  return res;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Store *op) {
  llvm::Value *array{nullptr};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array = GetVar(var_op->name);
  }
  CHECK(array) << "array is null";

  ir::Expr index = op->index();
  std::vector<llvm::Value *> indices;
  indices.push_back(Visit(&index));

  return Store(Visit(&op->value), GEP(array, std::move(indices)));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Alloc *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  auto *buffer    = GetVar(buffer_op->name);
  CHECK(buffer);

  return buffer;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Free *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  CHECK(named_vars_->count(buffer_op->name));
  named_vars_->erase(buffer_op->name);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Range_ *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::_IterVar_ *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::_Buffer_ *op) { return GetVar(op->name); }

llvm::Value *CodeGenLLVM::Visit(const ir::_Tensor_ *op) {
  auto *buffer_op = op->buffer.As<ir::_Buffer_>();
  CHECK(!named_vars_->count(buffer_op->name));
  return SetVar(buffer_op->name, Visit(buffer_op));
}

llvm::Value *CodeGenLLVM::Visit(const ir::_LoweredFunc_ *op) {
  CHECK_EQ(op->alloc_output_buffer_exprs.size(), op->dealloc_output_buffer_exprs.size())
      << "the count of allocation and deallocaton expressions is not match";

  std::vector<Expr> new_body;
  new_body.reserve(op->argument_prepare_exprs.size() + op->alloc_output_buffer_exprs.size() +
                   op->buffer_data_cast_exprs.size() + 1 /*op->body*/ + op->dealloc_output_buffer_exprs.size());

  auto new_body_append = [&new_body](auto &&... v) {
    auto append = [&new_body](auto &&v) {
      if constexpr (std::is_same<const ir::Expr &, decltype(v)>::value) {
        new_body.push_back(v);
      } else {
        new_body.insert(new_body.end(), v.begin(), v.end());
      }
    };
    (append(v), ...);
  };

  new_body_append(op->argument_prepare_exprs,
                  op->alloc_output_buffer_exprs,
                  op->buffer_data_cast_exprs,
                  op->body,
                  op->dealloc_output_buffer_exprs);

  ir::Expr function_body = ir::Block::Make(new_body);

  // Emit Function
  std::vector<llvm::Type *> arg_types = {b_->getInt8PtrTy(), b_->getInt32Ty()};

  llvm::FunctionType *function_type = llvm::FunctionType::get(
      /*Result=*/b_->getVoidTy(),
      /*Params=*/std::move(arg_types),
      /*isVarArg=*/false);
  CHECK(m_->getFunction(op->name) == nullptr) << "function[" << op->name << "] exists";

  llvm::Function *function = llvm::Function::Create(
      /*FunctionType=*/function_type,
      /*LinkageTypes=*/llvm::Function::ExternalLinkage,
      /*Name=*/op->name,
      /*Module=*/m_);
  function->setCallingConv(llvm::CallingConv::C);
  // function->addFnAttr("no-frame-pointer-elim", "false");
  function->setHasUWTable();  // GDB

  std::vector<llvm::Value *> args(function->arg_size());
  std::transform(
      function->arg_begin(), function->arg_end(), args.begin(), [](auto &arg) { return std::addressof(arg); });

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function,
      /*InsertBefore=*/nullptr);

  llvm::Value *old_args = GetVar("_args");  // store _args
  SetVar("_args", args[0]);
  b_->SetInsertPoint(entry);
  Visit(&function_body);
  if (old_args) {
    SetVar("_args", old_args);  // restore _args
  } else {
    named_vars_->erase("_args");
  }
  RetVoid();
  return function;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Let *op) {
  CHECK(op->type().valid());
  auto name = op->symbol.As<ir::_Var_>()->name;
  if (op->body.defined()) {
    SetVar(name, Visit(&op->body));
  } else {
    SetVar(name, Alloca(CinnTypeToIrType(op->type(), m_)));
  }
  return GetVar(name);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Reduce *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Ramp *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Broadcast *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::FracOp *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Power *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Product *op) {
  auto size = op->operands().size();
  if (size == 0) return nullptr;

  llvm::Value *ret = Visit(&op->operand(0));
  for (int i = 1; i < size; i++) {
    llvm::Value *v = Visit(&op->operand(i));
    if (is_integral_type(op->type())) {
      ret = Mul(ret, v);
    } else {
      ret = FMul(ret, v);
    }
  }

  return ret;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Sum *op) {
  auto size = op->operands().size();
  if (size == 0) return nullptr;

  llvm::Value *ret = Visit(&op->operand(0));
  for (int i = 1; i < size; i++) {
    llvm::Value *v = Visit(&op->operand(i));
    if (is_integral_type(op->type())) {
      ret = Add(ret, v);
    } else {  // float
      ret = FAdd(ret, v);
    }
  }

  return ret;
}

#undef __IR_EMITTER_NOT_IMPLEMENTED

void CodeGenLLVM::Compile(const lang::Module &module) {
  for (auto &fn : module.functions()) {
    Expr fn_expr(fn);
    Visit(&fn_expr);
  }
}

llvm::Value *CodeGenLLVM::EmitCall_buffer_create(const ir::Call *op) {
  CHECK_EQ(op->read_args.size(), 2UL);
  const ir::_Buffer_ *buffer_arg = op->read_args.front().as_buffer();
  CHECK(buffer_arg);
  return nullptr;
}

llvm::Value *CodeGenLLVM::EmitCall_buffer_malloc(const ir::Call *op) { return nullptr; }

llvm::Value *CodeGenLLVM::EmitCall_get_address(const ir::Call *op) {
  if (auto *read_var = op->read_args.front().as_var()) {
    return GetVar(read_var->name);
  }

  if (auto *read_buf = op->read_args.front().as_buffer()) {
    return GetVar(read_buf->name);
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::EmitCall_debug_info(const ir::Call *op) {
  auto callee = m_->getFunction(runtime::debug_log_repr);
  CHECK_GE(op->read_args.size(), 1UL);
  std::vector<llvm::Value *> args;
  for (auto &arg : op->read_args) {
    args.push_back(Visit(&arg));
  }
  return Call(callee, args, "call debug_info");
}

llvm::Value *CodeGenLLVM::GetVar(const std::string &name, bool lazy) {
  auto it = named_vars_->find(name);
  if (!lazy) {
    CHECK(it != named_vars_->end()) << "No var [" << name << "] found";
    return it->second;
  }
  return (*named_vars_)[name];
}

llvm::Value *CodeGenLLVM::SetVar(const std::string &name, llvm::Value *val) {
  (*named_vars_)[name] = val;
  CHECK(GetVar(name));
  return val;
}

}  // namespace backends
}  // namespace cinn
