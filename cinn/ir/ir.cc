#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

Expr Cast::Make(Type t, Expr v) {
  auto node = std::make_shared<Cast>(t, v);
  return Expr(node);
}

Expr Add::Make(Expr a, Expr b) {
  auto node = std::make_shared<Add>(a, b);
  return Expr(node);
}

Expr Sub::Make(Expr a, Expr b) {
  auto node = std::make_shared<Sub>(a, b);
  return Expr(node);
}

Expr Mul::Make(Expr a, Expr b) {
  auto node = std::make_shared<Mul>(a, b);
  return Expr(node);
}

Expr Div::Make(Expr a, Expr b) {
  auto node = std::make_shared<Div>(a, b);
  return Expr(node);
}

Expr Mod::Make(Expr a, Expr b) {
  auto node = std::make_shared<Mod>(a, b);
  return Expr(node);
}

Expr Min::Make(Expr a, Expr b) {
  auto node = std::make_shared<Min>(a, b);
  return Expr(node);
}

Expr Max::Make(Expr a, Expr b) {
  auto node = std::make_shared<Max>(a, b);
  return Expr(node);
}

Expr EQ::Make(Expr a, Expr b) {
  auto node = std::make_shared<EQ>(a, b);
  return Expr(node);
}

Expr NE::Make(Expr a, Expr b) {
  auto node = std::make_shared<NE>(a, b);
  return Expr(node);
}

Expr LT::Make(Expr a, Expr b) {
  auto node = std::make_shared<LT>(a, b);
  return Expr(node);
}

Expr LE::Make(Expr a, Expr b) {
  auto node = std::make_shared<LE>(a, b);
  return Expr(node);
}

Expr GT::Make(Expr a, Expr b) {
  auto node = std::make_shared<GT>(a, b);
  return Expr(node);
}

Expr GE::Make(Expr a, Expr b) {
  auto node = std::make_shared<GE>(a, b);
  return Expr(node);
}

Expr And::Make(Expr a, Expr b) {
  auto node = std::make_shared<And>(a, b);
  return Expr(node);
}

Expr Or::Make(Expr a, Expr b) {
  auto node = std::make_shared<Or>(a, b);
  return Expr(node);
}

Expr Not::Make(Expr v) {
  auto node = std::make_shared<Not>(v);
  return Expr(node);
}

Expr Variable::Make(const std::string &name, Type type) {
  auto node = std::make_shared<Variable>(name, type);
  return Expr(node);
}

For::For(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(body.defined());

  this->min = std::move(min);
  this->extent = std::move(extent);
  this->for_type = std::move(for_type);
  this->device_api = device_api;
  this->body = std::move(body);
}

Stmt For::Make(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body) {
  auto node = std::make_shared<For>(min, extent, for_type, device_api, body);
  return Stmt(node);
}

Stmt Block::Make(const std::vector<Stmt> &stmts) {
  auto node = std::make_shared<Block>();
  node->stmts = stmts;
  return Stmt(node);
}

Stmt IfThenElse::Make(Expr condition, Stmt true_case, Stmt false_case) {
  auto node = std::make_shared<IfThenElse>(condition, true_case, false_case);
  return Stmt(node);
}
}  // namespace ir
}  // namespace cinn
