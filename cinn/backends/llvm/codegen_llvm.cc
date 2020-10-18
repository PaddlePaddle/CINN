#include "cinn/backends/llvm/codegen_llvm.h"

#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <type_traits>

#include "cinn/backends/extern_func_emitter.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/common/cas.h"
#include "cinn/common/type.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"

namespace cinn {
namespace backends {

using BinaryInstruction = llvm::Instruction::BinaryOps;

namespace {

template <typename T>
auto NodeToExpr(const T *node) {
  std::ostringstream oss;
  // oss << "\033[32m";
  oss << ir::Expr(const_cast<T *>(node));
  // oss << "\033[0m";
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

  return comparison_result;
}

#define __IR_EMITTER_NOT_IMPLEMENTED(__op) CINN_NOT_IMPLEMENTED

int NextPowerOfTwo(int x) {
  for (int p2 = 1;; p2 *= 2) {
    if (p2 >= x) {
      return p2;
    }
  }
  return 0;
}

}  // namespace

CodeGenLLVM::CodeGenLLVM(llvm::Module *m, llvm::IRBuilder<> *b, const std::shared_ptr<SymbolTable> &symbol_table)
    : m_(m), b_(b), symbol_table_(symbol_table) {
  if (!symbol_table.get()) {
    symbol_table_ = std::make_shared<SymbolTable>();
  }
  symbol_table_->PushScope();  // Create a new scope by default.

  md_builder_        = std::make_unique<llvm::MDBuilder>(b_->getContext());
  md_tbaa_root_      = md_builder_->createTBAARoot("cinn-tbaa");
  md_tbaa_alias_set_ = md_builder_->createTBAANode("cinn-alias", md_tbaa_root_);

  InitTarget(common::DefaultHostTarget());
}

CodeGenLLVM::~CodeGenLLVM() {}

llvm::Value *CodeGenLLVM::EmitVectorSlice(llvm::Value *vec, int begin, int extent) {
  int numel = llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();
  if (extent == numel && begin == 0) return vec;

  CHECK(begin >= 0 && extent <= numel) << "Slicing out of bound!";

  std::vector<llvm::Constant *> indices(extent);
  for (int i = 0; i < extent; i++) {
    llvm::Constant **v = &indices[i];
    if (begin + i >= 0 && begin + i < numel) {
      *v = llvm::ConstantInt::get(b_->getInt32Ty(), begin + i);
    } else {
      *v = llvm::UndefValue::get(b_->getInt32Ty());
    }
  }
  return ShuffleVector(vec, vec, llvm::ConstantVector::get(std::move(indices)));
}

llvm::Value *CodeGenLLVM::EmitVectorPad(llvm::Value *vec, int lanes) {
#if LLVM_VERSION_MAJOR <= 10
  llvm::Value *mask = llvm::UndefValue::get(llvm::VectorType::get(b_->getInt32Ty(), lanes));
#else
  llvm::Value *mask =
      llvm::UndefValue::get(llvm::VectorType::get(b_->getInt32Ty(), llvm::ElementCount(lanes, false /*Scalable*/)));
#endif
  int numel = llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();

  CHECK(numel <= lanes);
  if (numel == lanes) return vec;
  for (int i = 0; i < numel; i++) {
    mask =
        InsertElement(mask, llvm::ConstantInt::get(b_->getInt32Ty(), i), llvm::ConstantInt::get(b_->getInt32Ty(), i));
  }

  return ShuffleVector(vec, vec, mask);
}

llvm::Value *CodeGenLLVM::EmitVectorConcat(std::vector<llvm::Value *> vecs) {
  int lanes = 0;
  for (auto *v : vecs) {
    lanes += llvm::dyn_cast<llvm::VectorType>(v->getType())->getNumElements();
  }
  while (vecs.size() > 1) {
    std::vector<llvm::Value *> new_vecs;
    for (size_t i = 0; i < vecs.size() - 1; i += 2) {
      auto *lhs            = vecs[i];
      auto *rhs            = vecs[i + 1];
      const auto lhs_lanes = llvm::dyn_cast<llvm::VectorType>(lhs->getType())->getNumElements();
      const auto rhs_lanes = llvm::dyn_cast<llvm::VectorType>(rhs->getType())->getNumElements();
      if (lhs_lanes < rhs_lanes) {
        lhs = EmitVectorPad(lhs, rhs_lanes);
      } else if (lhs_lanes > rhs_lanes) {
        rhs = EmitVectorPad(rhs, lhs_lanes);
      }

      const auto shared_lanes = std::max(lhs_lanes, rhs_lanes);
      std::vector<unsigned> mask(lhs_lanes + rhs_lanes);
      std::iota(mask.begin(), std::next(mask.begin(), lhs_lanes), 0);
      std::iota(std::next(mask.begin(), lhs_lanes), mask.end(), shared_lanes);
      new_vecs.push_back(ShuffleVector(lhs, rhs, mask));
    }
    if (vecs.size() % 2) {
      new_vecs.push_back(vecs.back());
    }

    vecs = std::move(new_vecs);
  }

  return EmitVectorSlice(vecs[0], 0, lanes);
}

llvm::Value *CodeGenLLVM::EmitBinaryOp(
    llvm::Value *lhs, llvm::Value *rhs, char opcode, bool is_integral, bool is_signed) {
  llvm::Instruction::BinaryOps ops;
  CHECK_EQ(lhs->getType(), rhs->getType())
      << "the types of operands of binary operation are mismatch"
      << ", lhs[" << DumpToString(*lhs) << "] " << opcode << " rhs[" << DumpToString(*rhs) << "]"
      << ", lhs_type[" << DumpToString(*lhs->getType()) << "], rhs_type[" << DumpToString(*rhs->getType()) << "]";
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
  if (op->type().is_bool()) {
    auto *type = b_->getInt8Ty();
    return llvm::ConstantInt::get(type, op->value, false);
  }
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
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());
  return EmitBinaryOp(lhs, rhs, '*', is_integral_type(op->type()));
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

  llvm::Type *source = CinnTypeToLLVMType(from, m_);
  llvm::Type *target = CinnTypeToLLVMType(to, m_);
  CHECK(source) << "source ir type is null";
  CHECK(target) << "target ir type is null";

  llvm::Value *value = Visit(&op->v());
  CHECK(value) << "value is null";

  // pod_value_t cast to a value.
  if (op->v().type().is_customized_type() &&
      op->v().type().customized_type() == common::customized_type::kpod_value_t) {  // pod_value_t operator
    llvm::Function *callee{};
    if (op->type().is_int(32)) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_int32);
    } else if (op->type().is_int(64)) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_int64);
    } else if (op->type().is_float(32)) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_float);
    } else if (op->type().is_float(64)) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_double);
    } else if (op->type() == type_of<void *>()) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_void_p);
    } else if (op->type() == type_of<cinn_buffer_t *>() || op->type() == type_of<const cinn_buffer_t *>()) {
      callee = m_->getFunction(runtime::intrisic::pod_value_to_buffer_p);
    } else {
      LOG(ERROR) << "can't cast cinn_pod_value_t to " << op->type();
      CINN_NOT_IMPLEMENTED
    }

    CHECK(callee);
    CHECK(op->v().as_var()) << "argument to the intrinsic function "
                               "cinn_pod_value_to_x should be a Var";
    value = GetVar(op->v().as_var()->name);
    return Call(callee, std::vector<llvm::Value *>({value}), "pod_value_cast");
  }

  do {
    if (value->getType() == target) break;

    if (to.is_cpp_handle() || to.is_cpp_handle_handle()) {
      value = BitCast(value, target, "cast_to_cpp_handle");
      break;
    }

    LOG(INFO) << "instr: " << DumpToString(*value);

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
  SymbolTableGuard symbol_table_guard(*symbol_table_);

  do {
    break;
    llvm::BasicBlock *preheader_bb = b_->GetInsertBlock();
    auto *for_begin = llvm::BasicBlock::Create(b_->getContext(), "for_begin", b_->GetInsertBlock()->getParent());
    auto *for_body  = llvm::BasicBlock::Create(b_->getContext(), "for_body", b_->GetInsertBlock()->getParent());
    auto *for_end   = llvm::BasicBlock::Create(b_->getContext(), "for_end", b_->GetInsertBlock()->getParent());

    Br(for_begin);
    b_->SetInsertPoint(for_begin);

    auto *begin      = Visit(&op->min);
    auto *loop_value = PHI(begin->getType(), 2);
    loop_value->addIncoming(begin, preheader_bb);

    llvm::Value *old_var = GetVar(op->loop_var->name);
    SetVar(op->loop_var->name, loop_value);
    auto *end = Visit(&op->extent);
    CondBr(ICmpSGE(loop_value, end), for_body, for_end);
    b_->SetInsertPoint(for_body);
    Visit(&op->body);

    if (old_var) {
      SetVar(op->loop_var->name, old_var);
    } else {
      symbol_table_->Erase(op->loop_var->name);
    }

    auto loop_next = Add(loop_value, llvm::ConstantInt::get(b_->getInt32Ty(), 1), "indvar.inc", true, true);
    loop_value->addIncoming(loop_next, b_->GetInsertBlock());

    Br(for_begin);
    b_->SetInsertPoint(for_end);

    return nullptr;
    // llvm::AllocaInst *loop_var = Alloca(b_->getInt32Ty(), nullptr, op->loop_var->name);
    // loop_var->setAlignment(llvm::Align(4));
    // SetVar(op->loop_var->name, loop_var);
  } while (false);

  ////////////////////////////////////
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
  loop_var->setAlignment(llvm::Align(4));
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
  llvm::BranchInst *back_branch = Br(header_bb);

  // Add loop metadata
  decltype(auto) ctx = b_->getContext();
  std::vector<llvm::Metadata *> loop_metadata;
  auto temp_node = llvm::MDNode::getTemporary(ctx, llvm::None);
  loop_metadata.push_back(temp_node.get());

  // TODO(fc500110): Loop vectorize
  // auto *vectorization = op->metadata.vectorization ? b_->getTrue() : b_->getFalse();
  // loop_metadata.push_back(llvm::MDNode::get(
  //        ctx, {llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
  //        llvm::ConstantAsMetadata::get(b_->getFalse())}));

  // Loop unroll
  std::string llvm_unroll_metadata{"llvm.loop.unroll."};
  switch (op->metadata.unroll_mode) {
    case ir::LLVMForLoopMeta::FullyUnroll:
      llvm_unroll_metadata += "full";
      break;
    case ir::LLVMForLoopMeta::NoUnroll:
      llvm_unroll_metadata += "disable";
      break;
    default:
      llvm_unroll_metadata += "enable";
  }

  /*
  loop_metadata.push_back(llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, llvm_unroll_metadata)}));
  auto loop_id = llvm::MDNode::get(ctx, loop_metadata);
  loop_id->replaceOperandWith(0, loop_id);
  back_branch->setMetadata(llvm::LLVMContext::MD_loop, loop_id);
   */

  if (old_var) {
    SetVar(op->loop_var->name, old_var);
  } else {
    symbol_table_->Erase(op->loop_var->name);
  }

  b_->SetInsertPoint(exit_bb);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::PolyFor *op) {
  CINN_NOT_IMPLEMENTED
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Select *op) {
  return Select(Visit(&op->condition), Visit(&op->true_value), Visit(&op->false_value));
}

llvm::Value *CodeGenLLVM::Visit(const ir::IfThenElse *op) {
  SymbolTableGuard symbol_table_guard(*symbol_table_);

  bool emit_else = op->false_case.defined();

  auto &ll_ctx      = b_->getContext();
  auto *ll_function = b_->GetInsertBlock()->getParent();

  llvm::Value *cond            = Visit(&op->condition);
  llvm::BasicBlock *then_block = llvm::BasicBlock::Create(ll_ctx, "if-then", ll_function);
  llvm::BasicBlock *end_block  = llvm::BasicBlock::Create(ll_ctx, "if-end", ll_function);

  if (op->false_case.defined()) {
    llvm::BasicBlock *else_block = llvm::BasicBlock::Create(ll_ctx, "if-else", ll_function);
    CondBr(cond, then_block, else_block);

    // true case
    b_->SetInsertPoint(then_block);
    Visit(&op->true_case);
    Br(end_block);

    // false case
    b_->SetInsertPoint(else_block);
    Visit(&op->false_case);
    Br(end_block);
  } else {
    CondBr(cond, then_block, end_block);
    b_->SetInsertPoint(then_block);
    Visit(&op->true_case);
    Br(end_block);
  }
  b_->SetInsertPoint(end_block);

  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Block *op) {
  // Create a new scope holding the temporary variables.
  SymbolTableGuard symbol_table_guard(*symbol_table_);

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

llvm::Value *CodeGenLLVM::Visit(const ir::PrimitiveNode *) { CINN_NOT_IMPLEMENTED return nullptr; }

llvm::Value *CodeGenLLVM::Visit(const ir::Call *op) {
  if (op->name == runtime::intrisic::buffer_create) {
  } else if (op->name == runtime::intrisic::get_address_repr) {
    return EmitCall_get_address(op);
  } else if (op->name == runtime::intrisic::debug_log_repr) {
    return EmitCall_debug_info(op);
  } else if (op->is_extern_call()) {
    auto emitter_id = ExternFuncID{backend_llvm_host, op->name.c_str()};
    auto *emitter   = ExternFunctionEmitterRegistry::Global().Lookup(emitter_id);
    if (emitter) {
      // CHECK(emitter) << "No extern function emitter called " << emitter_id;
      emitter->BindCodeGen(this);
      emitter->Emit(op);
      return extern_func_emit_res_;
    }
  }

  llvm::Function *callee = m_->getFunction(op->name);
  CHECK(callee) << "Unknown function referenced. [" << op->name << "]";

  std::vector<llvm::Value *> args;
  for (const auto &e : op->read_args) {
    auto *arg = Visit(&e);
    CHECK(arg) << "argument " << e << " is null";
    args.push_back(arg);
  }
  for (const auto &e : op->write_args) {
    auto *arg = Visit(&e);
    CHECK(arg) << "argument " << e << " is null";
    args.push_back(arg);
  }

  if (op->is_cinn_call()) {
    args[0] = BitCast(args[0], ll_void_p_ty(), "cast_to_void_p");
  } else if (op->is_intrinsic_call() && op->name == runtime::intrisic::args_construct_repr) {
    args[0] = BitCast(args[0], ll_cinn_pod_p_ty(), "cast_to_pod_p");
  }

  // Type cast statements in the head section of a CINN function.
  if (utils::Startswith(op->name, "cinn_pod_value_to_")) {
    CHECK_EQ(op->read_args.size(), 1UL);
    CHECK_EQ(op->write_args.size(), 0UL);

    // prepare the argument
    auto arg_load = op->read_args.front().As<ir::Load>();
    CHECK(arg_load->tensor.As<ir::Cast>()) << "get " << arg_load->tensor;
    auto _array_ptr       = arg_load->tensor.As<ir::Cast>()->v();
    auto array_ptr        = GetVar(_array_ptr.as_var()->name);
    auto cast_to_pod_arr  = BitCast(array_ptr, ll_cinn_pod_p_ty(), "cast_to_pod_arr");
    auto indice           = arg_load->index();
    auto *get_element_ptr = InBoundsGEP(ll_cinn_pod_ty(), cast_to_pod_arr, Visit(&indice), "");
    args.clear();
    args.push_back(get_element_ptr);

    return Call(callee, std::move(args), "cinn_pod_value_to_");
  }

  return Call(callee, std::move(args));
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Module_ *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::_Var_ *op) {
  llvm::Value *value = GetVar(op->name, false);
  llvm::Value *result{};
  CHECK(value) << "ir::_Var_[" << op->name << "]: value is null";
  // TODO(fc500110) hard coding
  if (LLVM_WillVarLowerAsPointer(op->name)) {
    result = value;
  } else if (value->getType()->isPointerTy()) {
    result = Load(value, op->name + "_load");
  } else {
    result = value;
  }

  return result;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Load *op) {
  llvm::Value *array{nullptr};
  bool is_alias{false};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array    = GetVar(var_op->name);
    is_alias = alias_vars_.count(const_cast<ir::_Var_ *>(var_op));
  } else {
    array = Visit(&op->tensor);
  }
  CHECK(array) << "fail to Visit Load node: " << Expr(const_cast<ir::Load *>(op));

  ir::Expr index = op->index();
  if (index.type().lanes() <= 1) {
    std::vector<llvm::Value *> indices;
    indices.push_back(Visit(&index));

    // auto load_inst = Load(InBoundsGEP(array, std::move(indices)));
    auto *load_inst = AlignedLoad(InBoundsGEP(array, std::move(indices)), llvm::MaybeAlign());
    /*
    if (is_alias) {
      llvm::MDNode *meta = md_builder_->createTBAANode("cinn-alias", md_tbaa_root_);
      load_inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
    }
     */
    if (auto *load_tensor = op->tensor.as_tensor()) {
      AddTbaaMetadata(load_inst, load_tensor->name, op->index());
    }

    {
      int alignment = op->type().bits();
      alignment     = 8;
      CHECK(alignment > 0);
      load_inst->setAlignment(llvm::Align(std::min(alignment, 8)));
    }

    // TODO(fc500110): tbaa AliasAnalysis
    // auto md_tbaa_root      = md_builder_->createTBAARoot("cinn-tbaa");
    // auto md_tbaa_alias_set = md_builder_->createTBAANode("cinn-alias", md_tbaa_root);
    // llvm::MDNode *meta     = md_tbaa_alias_set;
    // load_inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
    return load_inst;
  } else {  // vector load
    Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
    if (dense_strided_ramp.defined()) {
      CHECK(op->type().is_vector());

      llvm::Value *buffer = Visit(&op->tensor);
      return DenseVectorLoad(op);
    } else {
      LOG(FATAL) << "unsupported Ramp index " << op->index();
    }
  }
}

llvm::Value *CodeGenLLVM::Visit(const ir::Store *op) {
  llvm::Value *array{nullptr};
  bool is_alias{false};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array    = GetVar(var_op->name);
    is_alias = alias_vars_.count(const_cast<ir::_Var_ *>(var_op));
  }
  CHECK(array) << "array is null";

  ir::Expr index = op->index();

  if (op->type().is_scalar()) {
    std::vector<llvm::Value *> indices;
    indices.push_back(Visit(&index));

    // auto *store_inst = Store(Visit(&op->value), InBoundsGEP(array, std::move(indices)));
    auto *store_inst = AlignedStore(Visit(&op->value), InBoundsGEP(array, std::move(indices)), llvm::MaybeAlign());
    /*
    if (is_alias) {
      llvm::MDNode *meta = md_builder_->createTBAANode("cinn-alias", md_tbaa_root_);
      store_inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
    }
     */
    {
      int alignment = op->type().bits();
      alignment     = 8;
      CHECK_GT(alignment, 0);
      store_inst->setAlignment(llvm::Align(std::min(alignment, 8)));
    }
    // TODO(fc500110): tbaa AliasAnalysis
    // auto md_tbaa_root      = md_builder_->createTBAARoot("cinn-tbaa");
    // auto md_tbaa_alias_set = md_builder_->createTBAANode("cinn-alias", md_tbaa_root);
    // llvm::MDNode *meta     = md_tbaa_alias_set;
    // store_inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
    AddTbaaMetadata(store_inst, op->tensor.as_tensor()->name, op->index());
    return store_inst;
  } else {  // vector store
    Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
    auto ramp_expr          = op->index();
    auto *ramp              = index.As<ir::Ramp>();
    if (dense_strided_ramp.defined()) {  // stride 1
      int alignment = op->type().ElementOf().bits();

      int native_bits  = 512;
      int native_bytes = native_bits / 8;

      alignment = 64;

      int total_lanes = op->type().lanes();
      int step        = native_bits / op->type().ElementOf().bits();

      auto *buffer = Visit(&op->tensor);
      auto *value  = Visit(&op->value);

      // fit the total_lanes in native_lanes(split into multiple native steps)
      for (int offset = 0; offset < total_lanes; offset += step) {
        int lanes = std::min(step, total_lanes - offset);
        Expr base = common::AutoSimplify(ramp->base + offset);
        auto *ptr = CreateBufferPtr(op->type().ElementOf(), buffer, Visit(&base));
        auto *vtype =
            llvm::VectorType::get(ll_type_of(op->type().ElementOf()), llvm::ElementCount(lanes, false /*Scalable*/))
                ->getPointerTo();
        llvm::StoreInst *inst =
            b_->CreateAlignedStore(CreateVecSlice(value, offset, lanes), b_->CreatePointerCast(ptr, vtype), alignment);
        AddTbaaMetadata(inst, op->tensor.as_tensor()->name, base);
        return inst;
      }
    }
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Alloc *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  auto *buffer    = GetVar(buffer_op->name);
  CHECK(buffer);

  return buffer;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Free *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  CHECK(symbol_table_->Lookup(buffer_op->name));
  symbol_table_->Erase(buffer_op->name);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Buffer_ *op) { return GetVar(op->name); }

llvm::Value *CodeGenLLVM::Visit(const ir::_Tensor_ *op) {
  return GetVar(op->name);
  auto *buffer_op = op->buffer.As<ir::_Buffer_>();
  // return (*named_vars_)[buffer_op->name] = Visit(buffer_op);
  if (symbol_table_->Lookup(buffer_op->name)) {
    return Visit(buffer_op);
  }

  return SetVar(buffer_op->name, Visit(buffer_op));
}

llvm::Value *CodeGenLLVM::Visit(const ir::_LoweredFunc_ *op) {
  auto init_function_state = [this]() { alias_vars_.clear(); };
  init_function_state();

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

  std::vector<llvm::Value *> args;
  args.reserve(function->arg_size());
  std::transform(function->arg_begin(), function->arg_end(), std::back_inserter(args), [](auto &arg) {
    return std::addressof(arg);
  });

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function,
      /*InsertBefore=*/nullptr);

  SetVar("_args", args[0]);
  b_->SetInsertPoint(entry);
  Visit(&function_body);
  symbol_table_->Erase("_args");
  RetVoid();

  return function;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Let *op) {
  CHECK(op->type().valid());
  auto name = op->symbol.As<ir::_Var_>()->name;
  if (op->symbol.As<ir::_Var_>()->type().is_cpp_handle()) {
    alias_vars_.insert(const_cast<ir::_Var_ *>(op->symbol.As<ir::_Var_>()));
  }
  if (op->body.defined()) {
    SetVar(name, Visit(&op->body));
  } else {
    llvm::AllocaInst *inst = Alloca(CinnTypeToLLVMType(op->type(), m_), nullptr, name);
    auto get_align         = [](int n) {
      int i{0}, r{1};
      while (n > r) {
        r *= 2;
        ++i;
      }
      return r / 8;
    };
    int align_bits = std::max<int>(op->type().bits(), 8);
    int align      = get_align(align_bits);
    inst->setAlignment(llvm::Align(align));
    SetVar(name, inst);
  }

  return GetVar(name);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Reduce *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Ramp *op) { __IR_EMITTER_NOT_IMPLEMENTED(op); }

llvm::Value *CodeGenLLVM::Visit(const ir::Broadcast *op) {
#if LLVM_VERSION_MAJOR >= 11
  const llvm::ElementCount elem_count(op->lanes, /*scalable*/ false);
#else
  const int elem_count = op->lanes;
#endif
  llvm::Value *value    = Visit(&op->value);
  llvm::Constant *undef = llvm::UndefValue::get(llvm::VectorType::get(value->getType(), elem_count));
  llvm::Constant *zero  = llvm::ConstantInt::get(ll_int32_ty(), 0);
  value                 = b_->CreateInsertElement(undef, value, zero, "broadcast");
  llvm::Constant *zeros = llvm::ConstantVector::getSplat(elem_count, zero);
  return b_->CreateShuffleVector(value, undef, zeros, "broadcast_shuffle");
}

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

#undef __IR_EMITTER_CINN_NOT_IMPLEMENTED

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
  auto callee = m_->getFunction(runtime::intrisic::debug_log_repr);
  CHECK_GE(op->read_args.size(), 1UL);
  std::vector<llvm::Value *> args;
  for (auto &arg : op->read_args) {
    args.push_back(Visit(&arg));
  }
  return Call(callee, args, "call debug_info");
}

llvm::Value *CodeGenLLVM::GetVar(const std::string &name, bool lazy) {
  auto symbol = symbol_table_->Lookup(name);
  if (!lazy) {
    CHECK(symbol) << "No var [" << name << "] found";
  }
  return symbol;
}

llvm::Value *CodeGenLLVM::SetVar(const std::string &name, llvm::Value *val) {
  symbol_table_->Insert(name, val);
  CHECK(GetVar(name));
  return val;
}

llvm::FunctionType *CodeGenLLVM::GenFunctionTypeFromCinnFunction(const ir::_LoweredFunc_ *func, bool with_buffer_type) {
  auto func_ret_type = CinnTypeToLLVMType(Void(), m_);
  std::vector<llvm::Type *> arg_types;
  for (auto &arg : func->args) {
    if (arg.is_buffer() && arg.is_var()) {
      alias_vars_.insert(arg.var_arg().get());
    }
    if (arg.is_var()) {
      arg_types.push_back(CinnTypeToLLVMType(arg.var_arg()->type(), m_));
    } else if (arg.is_buffer()) {
      if (with_buffer_type) {
        arg_types.push_back(ll_cinn_buffer_p_ty());
      } else {
        arg_types.push_back(CinnTypeToLLVMType(arg.buffer_arg()->type(), m_));
      }
    }
  }

  return llvm::FunctionType::get(func_ret_type, arg_types, false);
}

llvm::Value *CodeGenLLVM::DenseVectorLoad(const ir::Load *op) {
  auto index = op->index();
  auto *ramp = index.As<ir::Ramp>();
  CHECK(ramp);

  int alignment = op->type().bits();

  // for x86_64
  // TODO(Superjomn) replae this with arch.
  int native_bits  = 512;
  int native_bytes = native_bits / 8;

  alignment = 64;

  int load_lanes   = op->type().lanes();
  int native_lanes = native_bits / op->type().bits();

  std::vector<llvm::Value *> slices;

  llvm::Value *buffer = Visit(&op->tensor);
  buffer->setName("buffer");

  for (int i = 0; i < load_lanes; i += native_lanes) {
    int slice_lanes   = std::min(native_lanes, load_lanes - i);
    auto slice_base   = common::AutoSimplify(ramp->base + i);
    auto slide_stride = Expr(1);
    auto slide_index  = slice_base;

#if LLVM_VERSION_MAJOR >= 11
    const llvm::ElementCount elem_count(slice_lanes, /*scalable*/ false);
#else
    const int elem_count = slice_lanes;
#endif

    llvm::Type *slice_type = llvm::VectorType::get(CinnTypeToLLVMType(op->type().ElementOf(), m_), elem_count);

    llvm::Value *elt_ptr = CreateBufferPtr(op->type().ElementOf(), buffer, Visit(&slice_base));
    llvm::Value *vec_ptr = b_->CreatePointerCast(elt_ptr, slice_type->getPointerTo(), "get_vec_ptr");

    llvm::Instruction *load_inst = b_->CreateAlignedLoad(vec_ptr, llvm::Align(alignment), "load_vec");
    AddTbaaMetadata(load_inst, op->tensor.as_tensor()->name, op->index());

    slices.push_back(load_inst);
  }

  CHECK_EQ(slices.size(), 1UL);

  return slices[0];
}

llvm::Value *CodeGenLLVM::CreateBufferVecPtr(Type t, llvm::Value *buffer, llvm::Value *index) {
  CHECK_GT(t.lanes(), 1) << "type is not a vector type: " << t;
  llvm::PointerType *btype = llvm::dyn_cast<llvm::PointerType>(buffer->getType());
  CHECK(btype);
  llvm::PointerType *ptype = CinnTypeToLLVMType(t, m_)->getPointerTo(btype->getAddressSpace());
  if (btype != ptype) {
    buffer = b_->CreatePointerCast(buffer, ptype);
  }
  return b_->CreateInBoundsGEP(buffer, index);
}

llvm::Value *CodeGenLLVM::CreateBufferPtr(Type t, llvm::Value *buffer, llvm::Value *index) {
  CHECK_EQ(t.lanes(), 1);
  auto *btype = llvm::dyn_cast<llvm::PointerType>(buffer->getType());
  CHECK(btype);
  auto *ptype = CinnTypeToLLVMType(t, m_)->getPointerTo(btype->getAddressSpace());
  CHECK(ptype);
  if (btype != ptype) {
    buffer = b_->CreatePointerCast(buffer, ptype, "pointer_cast");
  }
  return b_->CreateInBoundsGEP(buffer, index, "buffer_ptr");
}

llvm::Value *CodeGenLLVM::CreateVecSlice(llvm::Value *vec, int begin, int lanes) {
  LOG(INFO) << "type: " << DumpToString(*vec->getType());
  int total_lanes = llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();
  CHECK_LE(begin + lanes, total_lanes);
  if (lanes == total_lanes && begin == 0) return vec;  // full slice
  std::vector<llvm::Constant *> indices;
  for (int i = 0; i < lanes; ++i) {
    indices.push_back(ll_const_int32(begin + i));
  }
  llvm::Constant *undef = llvm::UndefValue::get(vec->getType());
  return b_->CreateShuffleVector(vec, undef, llvm::ConstantVector::get(indices));
}

void CodeGenLLVM::InitTarget(const Target &target) {
  switch (target.arch) {
    case Target::Arch::X86:
      if (target.bits == Target::Bit::k32) {
        naive_vec_alignment_ = 256;
      } else if (target.bits == Target::Bit::k64) {
        naive_vec_alignment_ = 512;
      } else {
        LOG(FATAL) << "get unknown bits";
      }
      break;
    case Target::Arch::ARM:
      naive_vec_alignment_ = 128;
      break;
    case Target::Arch::NVGPU:
      naive_vec_alignment_ = 128;
      break;
    case Target::Arch::Unk:
      LOG(FATAL) << "unknown Arch found";
      break;
  }
}

bool LLVM_WillVarLowerAsPointer(const std::string &var_name) {
  return var_name == "_args" || utils::Endswith(var_name, "__ptr");
}

void CodeGenLLVM::AddTbaaMetadata(llvm::Instruction *inst, std::string_view buffer, Expr index) {
  // If the index is constant, generate some TBAA info that helps LLVM understand our loads/stores aren't aliased.
  bool constant_index = false;
  int base            = 0;
  int width           = 1;

  if (index.defined()) {
    if (const ir::Ramp *ramp = index.As<ir::Ramp>()) {
      auto *pstride_int = ramp->stride.As<ir::IntImm>();
      auto *pbase_int   = ramp->base.As<ir::IntImm>();
      if (pstride_int && pbase_int) {
        int stride = pstride_int->value;
        base       = pbase_int->value;
        CHECK_GE(base, 0);
        width = NextPowerOfTwo(ramp->lanes * stride);

        while (base % width) {
          base -= base % width;
          width *= 2;
        }
        constant_index = true;
      }
    } else {
      auto *pbase_int = index.As<ir::IntImm>();
      if (pbase_int) {
        int pbase      = pbase_int->value;
        base           = pbase;
        constant_index = true;
      }
    }
  }

  llvm::MDBuilder builder(b_->getContext());

  // Add type-based-alias-analysis metadata to the pointer, so that loads and stores to different buffers can get
  // reordered.
  llvm::MDNode *tbaa = builder.createTBAARoot("cinn buffer");
  tbaa               = builder.createTBAAScalarTypeNode(std::string(buffer), tbaa);

  // Add metadata fro constant indices to allow loads and stores to the same buffer to get reordered.
  if (constant_index) {
    for (int w = 1024; w >= width; w /= 2) {
      int b = (base / w) * w;
      tbaa  = builder.createTBAAScalarTypeNode(utils::StringFormat("%s.width%d.base%d", buffer.data(), w, b), tbaa);
    }
  }

  tbaa = builder.createTBAAStructTagNode(tbaa, tbaa, 0);
  inst->setMetadata("tbaa", tbaa);
}

}  // namespace backends
}  // namespace cinn
