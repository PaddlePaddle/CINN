#include "cinn/ir/ir_printer.h"

#include <algorithm>
#include <vector>

#include "cinn/ir/lowered_func.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/module.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

void IrPrinter::Print(Expr e) { IRVisitor::Visit(&e); }
void IrPrinter::Print(const std::vector<Expr> &exprs, const std::string &splitter) {
  for (int i = 0; !exprs.empty() && i < exprs.size() - 1; i++) {
    Print(exprs[i]);
    os_ << splitter;
  }
  if (!exprs.empty()) Print(exprs.back());
}

void IrPrinter::Visit(const IntImm *x) { os_ << x->value; }
void IrPrinter::Visit(const UIntImm *x) { os_ << x->value; }
void IrPrinter::Visit(const FloatImm *x) { os_ << x->value; }
void IrPrinter::Visit(const StringImm *x) { os_ << "\"" << x->value << "\""; }
void IrPrinter::Visit(const Add *x) { PrintBinaryOp("+", x); }
void IrPrinter::Visit(const Sub *x) { PrintBinaryOp("-", x); }
void IrPrinter::Visit(const Mul *x) { PrintBinaryOp("*", x); }
void IrPrinter::Visit(const Div *x) { PrintBinaryOp("/", x); }
void IrPrinter::Visit(const Mod *x) { PrintBinaryOp("%", x); }
void IrPrinter::Visit(const EQ *x) { PrintBinaryOp("==", x); }
void IrPrinter::Visit(const NE *x) { PrintBinaryOp("!=", x); }
void IrPrinter::Visit(const LT *x) { PrintBinaryOp("<", x); }
void IrPrinter::Visit(const LE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(const GT *x) { PrintBinaryOp(">", x); }
void IrPrinter::Visit(const GE *x) { PrintBinaryOp(">=", x); }
void IrPrinter::Visit(const And *x) { PrintBinaryOp("and", x); }
void IrPrinter::Visit(const Or *x) { PrintBinaryOp("or", x); }
void IrPrinter::Visit(const Not *x) {
  os_ << "!";
  Print(x->v());
}
void IrPrinter::Visit(const Min *x) {
  os_ << "cinn_min(";
  Print(x->a());
  os_ << ", ";
  Print(x->b());
  os_ << ")";
}
void IrPrinter::Visit(const Max *x) {
  os_ << "cinn_max(";
  Print(x->a());
  os_ << ", ";
  Print(x->b());
  os_ << ")";
}
void IrPrinter::Visit(const Minus *x) {
  os_ << "-(";
  Print(x->v());
  os_ << ")";
}
void IrPrinter::Visit(const For *x) {
  os_ << "for (";
  Print(x->loop_var);
  os_ << ", ";
  Print(x->extent);
  os_ << ")\n";

  DoIndent();
  Print(x->body);
}

void IrPrinter::Visit(const PolyFor *x) {
  os_ << "poly_for (";
  Print(x->iterator);
  os_ << ", ";
  Print(x->init);
  os_ << ", ";
  Print(x->condition);
  os_ << ", ";
  Print(x->inc);
  os_ << ")\n";

  DoIndent();
  Print(x->body);
}
void IrPrinter::Visit(const IfThenElse *x) {
  os_ << "if (";
  Print(x->condition);
  os_ << ") {\n";
  IncIndent();
  DoIndent();
  Print(x->true_case);
  DecIndent();
  os() << "\n";
  DoIndent();
  os() << "}";

  if (x->false_case.defined()) {
    os_ << " else {\n";
    IncIndent();

    DoIndent();
    Print(x->false_case);
    os() << "\n";

    DecIndent();
    DoIndent();
    os_ << "}";
  }
}
void IrPrinter::Visit(const Block *x) {
  os_ << "{\n";

  IncIndent();
  for (int i = 0; i < x->stmts.size() - 1; i++) {
    DoIndent();
    Print(x->stmts[i]);
    os_ << "\n";
  }
  if (x->stmts.size() >= 1) {
    DoIndent();
    Print(x->stmts.back());
  }
  DecIndent();
  os_ << "\n";
  DoIndent();
  os_ << "}";
}
void IrPrinter::Visit(const Call *x) {
  os_ << x->name << "(";
  if (!x->read_args.empty()) {
    for (int i = 0; i < x->read_args.size() - 1; i++) {
      Print(x->read_args[i]);
      os_ << ", ";
    }
    Print(x->read_args.back());
  }

  if (!x->write_args.empty()) {
    if (!x->read_args.empty()) os() << ", ";

    for (int i = 0; i < x->write_args.size() - 1; i++) {
      Print(x->write_args[i]);
      os_ << ", ";
    }
    Print(x->write_args.back());
  }

  os_ << ")";
}
void IrPrinter::Visit(const Cast *x) {
  os() << x->type();
  os() << "(";
  os() << x->v();
  os() << ")";
}
void IrPrinter::Visit(const _Module_ *x) {}
void IrPrinter::Visit(const _Var_ *x) { os_ << x->name; }
void IrPrinter::Visit(const Alloc *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  os_ << "alloc(" << buffer->name << ", ";
  Print(x->extents);
  os_ << ")";
}
void IrPrinter::Visit(const Select *x) {
  os_ << "select(";
  Print(x->condition);
  os_ << ", ";
  Print(x->true_value);
  os_ << ", ";
  Print(x->false_value);
  os_ << ")";
}
void IrPrinter::Visit(const Load *x) {
  if (x->is_addr_tensor()) {
    auto *tensor = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor);
    os_ << tensor->name;
  } else if (x->is_addr_scalar()) {
    Print(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  os_ << "[";
  for (int i = 0; i < x->indices.size() - 1; i++) {
    Print(x->indices[i]);
    os() << ", ";
  }
  if (!x->indices.empty()) Print(x->indices.back());
  os_ << "]";
}
void IrPrinter::Visit(const Store *x) {
  if (x->is_addr_tensor()) {
    auto *tensor_node = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor_node);
    os_ << tensor_node->name;
  } else if (x->is_addr_scalar()) {
    Print(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  os_ << "[";
  for (int i = 0; i < x->indices.size() - 1; i++) {
    Print(x->indices[i]);
    os() << ", ";
  }
  if (!x->indices.empty()) Print(x->indices.back());
  os_ << "] = ";
  Print(x->value);
}
void IrPrinter::Visit(const Free *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  os_ << "free(" << buffer->name << ")";
}

void IrPrinter::DoIndent() { os_ << std::string(indent_, ' '); }
void IrPrinter::IncIndent() { indent_ += indent_unit; }
void IrPrinter::DecIndent() { indent_ -= indent_unit; }

void IrPrinter::Visit(const _Range_ *x) {
  os_ << "Range(min=";
  Print(x->min);
  os_ << ", "
      << "extent=";
  Print(x->extent);
  os_ << ")";
}

void IrPrinter::Visit(const _Buffer_ *x) {
  std::vector<std::string> dim_names;
  std::transform(x->shape.begin(), x->shape.end(), std::back_inserter(dim_names), [&](const Expr &x) {
    return utils::GetStreamCnt(x);
  });

  os_ << "_Buffer_<" << x->type() << ": " << utils::Join(dim_names, ",") << ">(" << x->name << ")";
}
void IrPrinter::Visit(const _Tensor_ *x) {
  CHECK(!x->shape.empty());
  os_ << "Tensor(";
  os() << x->name << ", ";
  os() << "[";
  if (!x->shape.empty()) {
    for (int i = 0; i < x->shape.size() - 1; i++) {
      Print(x->shape[i]);
      os() << ",";
    }
    Print(x->shape.back());
  }
  os_ << "])";
}
void IrPrinter::Visit(const _LoweredFunc_ *f) {
  os_ << "function " << f->name << " ";

  std::vector<std::string> arg_names;
  for (auto &arg : f->args) {
    arg_names.push_back(arg.name());
  }
  os_ << "(" << utils::Join(arg_names, ", ") << ")\n";

  Print(f->body);
}
void IrPrinter::Visit(const Let *f) {
  CHECK(f->type().valid());
  os() << f->type() << " ";
  Print(f->symbol);
  if (f->body.defined()) {
    os() << " = ";
    Print(f->body);
  }
}

void IrPrinter::Visit(const Reduce *f) {
  os() << "Reduce(";
  switch (f->reduce_type) {
    case Reduce::ReduceType::kSum:
      os() << "sum";
      break;
    case Reduce::ReduceType::kSub:
      os() << "sub";
      break;
    case Reduce::ReduceType::kDiv:
      os() << "Div";
      break;
    case Reduce::ReduceType::kMul:
      os() << "Mul";
      break;
    case Reduce::ReduceType::kMax:
      os() << "Max";
      break;
    case Reduce::ReduceType::kMin:
      os() << "Min";
      break;
  }
  os() << ", ";
  Print(f->body);
  os() << ",";
  Print(f->init);
  os() << ")";
}

void IrPrinter::Visit(const Ramp *x) {
  os() << "Ramp(";
  Print(x->base);
  os() << ",";
  Print(x->stride);
  os() << ",";
  os() << x->lanes;
  os() << ")";
}

void IrPrinter::Visit(const Broadcast *x) {
  os() << "Broadcast(";
  Print(x->value);
  os() << ",";
  os() << x->lanes;
  os() << ")";
}

void IrPrinter::Visit(const FracOp *x) {
  os() << "(";
  Print(x->a());
  os() << " / ";
  Print(x->b());
  os() << ")";
}

void IrPrinter::Visit(const Power *x) {
  os() << "(";
  Print(x->a());
  os() << "^";
  Print(x->b());
  os() << ")";
}

void IrPrinter::Visit(const Product *x) {
  os() << "(";
  for (int i = 0; i < x->operands().size() - 1; i++) {
    Print(x->operand(i));
    os() << " * ";
  }
  if (!x->operands().empty()) Print(x->operands().back());
  os() << ")";
}

void IrPrinter::Visit(const Sum *x) {
  os() << "(";
  for (int i = 0; i < x->operands().size() - 1; i++) {
    Print(x->operand(i));
    os() << " + ";
  }
  if (!x->operands().empty()) Print(x->operands().back());
  os() << ")";
}

void IrPrinter::Visit(const PrimitiveNode *x) {
  os() << x->name << "(";
  std::vector<std::string> args_repr;
  for (auto &args : x->arguments) {
    std::vector<std::string> arg_repr;
    for (auto &arg : args) {
      arg_repr.push_back(utils::GetStreamCnt(arg));
    }
    args_repr.push_back(utils::Join(arg_repr, ","));
  }

  os() << utils::Join(args_repr, ",");
  os() << ")";
}

std::ostream &operator<<(std::ostream &os, Expr a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<Expr> &a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const lang::Module &m) {
  os << "Module " << m->name << " {\n\n";

  for (auto &fn : m->functions) {
    os << fn << '\n';
  }

  os << "\n\n}";

  return os;
}

}  // namespace ir
}  // namespace cinn
