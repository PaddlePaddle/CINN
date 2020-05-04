#include "cinn/optim/insert_debug_log_callee.h"

#include <sstream>
#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {
using cinn::utils::StringFormat;

namespace {

struct InsertDebugLogCalleeMutator : public ir::IRMutator<> {
  void operator()(Expr* e) { ir::IRMutator<>::Visit(e, e); }

  void Visit(const ir::_LoweredFunc_* op, Expr* expr) {
    auto* node       = expr->As<ir::_LoweredFunc_>();
    auto* body_block = node->body.As<ir::Block>();
    CHECK(body_block);

    auto msg        = StringFormat("running : %s", GetDebugString(*expr).c_str());
    auto debug_node = CreateDebugStatement(msg);

    ir::IRMutator<>::Visit(&node->body, &node->body);

    auto deal_with_exprs = [&](std::vector<Expr>* exprs) {  // deal with op->argument_preapre_exprs
      std::vector<Expr> new_stmts;
      for (auto& expr : *exprs) {
        auto msg = StringFormat("running : %s", GetDebugString(expr).c_str());
        new_stmts.push_back(CreateDebugStatement(msg));
        new_stmts.push_back(expr);
      }
      *exprs = new_stmts;
    };

    deal_with_exprs(&node->alloc_output_buffer_exprs);
    deal_with_exprs(&node->dealloc_output_buffer_exprs);
    deal_with_exprs(&node->buffer_data_cast_exprs);
    deal_with_exprs(&node->argument_prepare_exprs);

    body_block->stmts.insert(body_block->stmts.begin(), debug_node);
  }

  void Visit(const ir::Block* op, Expr* expr) {
    auto* node = expr->As<ir::Block>();
    std::vector<Expr> new_stmts;
    for (auto& e : op->stmts) {
      if (!IsDebugInfoNode(e)) {
        auto msg             = StringFormat("running: %s", GetDebugString(e).c_str());
        auto debug_info_node = CreateDebugStatement(msg);
        new_stmts.push_back(debug_info_node);
      }

      ir::IRMutator<>::Visit(&e, &Reference(&e));
      new_stmts.push_back(e);
    }

    node->stmts = new_stmts;
  }

  std::string GetDebugString(const Expr& e) {
    std::stringstream ss;
    switch (e.node_type()) {
      case ir::IrNodeTy::Block:
        ss << "<block>";
        break;
      case ir::IrNodeTy::For: {
        auto* node = e.As<ir::For>();
        ss << "<For " << node->loop_var << " in [" << node->min << ", " << node->extent << ")>";
        break;
      }
      case ir::IrNodeTy::PolyFor: {
        auto* node = e.As<ir::PolyFor>();
        ss << "<PolyFor " << node->iterator << " in [" << node->init << ", " << node->extent() << ")"
           << " with condition: " << node->condition << ">";
        break;
      }
      case ir::IrNodeTy::_LoweredFunc_: {
        auto* node = e.As<ir::_LoweredFunc_>();
        ss << "<LoweredFunc " << node->name << ">";
        break;
      }
      case ir::IrNodeTy::Store: {
        ss << e;
        break;
      }
      case ir::IrNodeTy::Call: {
        auto* node = e.As<ir::Call>();
        if (node->name == runtime::debug_log_repr) {
          return "";
        }
        ss << e;
        break;
      }
      default:
        ss << "NodeTy " << e->node_type() << ": " << e;
        break;
    }

    return ss.str();
  }

  inline bool IsDebugInfoNode(const Expr& e) {
    return e.As<ir::Call>() && e.As<ir::Call>()->name == runtime::debug_log_repr;
  }

  Expr CreateDebugStatement(const std::string& msg) {
    return ir::Call::Make(Void(), runtime::debug_log_repr, {Expr(msg)}, {}, ir::Call::CallType ::Intrinsic);
  }
};

}  // namespace

void InsertDebugLogCallee(Expr* e) { InsertDebugLogCalleeMutator()(e); }

}  // namespace optim
}  // namespace cinn
