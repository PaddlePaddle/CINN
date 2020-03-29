#include "cinn/common/ir.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace common {

namespace {

// ramp + scalar or broadcast
Expr RampRelatedMul(ir::Ramp *ramp, Expr other) {
  CHECK_EQ(other.type().ElementOf(), Int(32));
  CHECK_EQ(ramp->base.type(), Int(32));
  CHECK_EQ(ramp->stride.type(), Int(32));
  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    CHECK_EQ(ramp->lanes, other_broadcast->lanes);
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base, ramp->stride * other, ramp->lanes);
}

Expr RampRelatedMul(ir::Broadcast *broadcast, Expr other) {
  CHECK_EQ(other.type().lanes(), 1);
  return ir::Broadcast::Make(broadcast->value * other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedMul(ir::Ramp *ramp, ir::Ramp *other) {
  NOT_IMPLEMENTED
  return Expr();
}
// ramp + scalar
Expr RampRelatedAdd(ir::Ramp *ramp, Expr other) {
  CHECK_EQ(other.type().ElementOf(), Int(32));

  auto *other_broadcast = other.As<ir::Broadcast>();
  if (other_broadcast) {
    CHECK_EQ(ramp->lanes, other_broadcast->lanes);
    other = other_broadcast->value;
  }
  return ir::Ramp::Make(ramp->base + other, ramp->stride, ramp->lanes);
}
Expr RampRelatedAdd(ir::Broadcast *broadcast, Expr other) {
  CHECK_EQ(other.type().lanes(), 1);
  return ir::Broadcast::Make(broadcast->value + other, broadcast->lanes);
}
// ramp * ramp
Expr RampRelatedAdd(ir::Ramp *ramp, ir::Ramp *other) {
  NOT_IMPLEMENTED
  return Expr();
}

Expr RampRelatedAdd(Expr a, Expr b) {
  auto *a_ramp      = a.As<ir::Ramp>();
  auto *b_ramp      = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (b->type().lanes() == 1 || b_broadcast)) {
    return RampRelatedAdd(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().lanes() == 1 || a_broadcast)) {
    return RampRelatedAdd(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() && !b->type().is_vector()) {
    return a + b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedAdd(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedAdd(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedAdd(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    CHECK_EQ(a_broadcast->lanes, b_broadcast->lanes);
    return ir::Broadcast::Make(a_broadcast->value + b_broadcast->value, a_broadcast->lanes);
  } else {
    NOT_IMPLEMENTED
  }
}

Expr RampRelatedMul(Expr a, Expr b) {
  auto *a_ramp      = a.As<ir::Ramp>();
  auto *b_ramp      = b.As<ir::Ramp>();
  auto *a_broadcast = a.As<ir::Broadcast>();
  auto *b_broadcast = b.As<ir::Broadcast>();
  if (a_ramp && !b_ramp && (!b->type().is_vector() || b_broadcast)) {
    return RampRelatedMul(a_ramp, b);
  } else if (!a_ramp && b_ramp && (a->type().is_vector() || a_broadcast)) {
    return RampRelatedMul(b_ramp, a);
  } else if (!a_ramp && !b_ramp && !a->type().is_vector() && !b->type().is_vector()) {
    return a * b;
  } else if (a_ramp && b_ramp) {  // a_ramp && b_ramp
    return RampRelatedMul(a_ramp, b_ramp);
  } else if (a_broadcast && !b_broadcast) {
    return RampRelatedMul(a_broadcast, b);
  } else if (!a_broadcast && b_broadcast) {
    return RampRelatedMul(b_broadcast, a);
  } else if (a_broadcast && b_broadcast) {
    CHECK_EQ(a_broadcast->lanes, b_broadcast->lanes);
    return ir::Broadcast::Make(a_broadcast->value * b_broadcast->value, a_broadcast->lanes);
  } else {
    LOG(INFO) << "a,b: " << a << " " << b;
    NOT_IMPLEMENTED
  }
}

}  // namespace

Expr ExpandTo1DIndice(const std::vector<Expr> &shape, const std::vector<Expr> &indices) {
  CHECK_EQ(shape.size(), indices.size());
  Expr res;
  for (int i = 0; i < shape.size(); i++) {
    CHECK_EQ(shape[i].type(), Int(32));
    Expr indice_prod = indices[i];
    for (int j = i + 1; j < shape.size(); j++) {
      indice_prod = RampRelatedMul(indice_prod, shape[j]);
    }

    if (res.defined()) {
      res = RampRelatedAdd(res, indice_prod);
    } else {
      res = indice_prod;
    }
  }

  return res;
}

Expr ExpandTo1DIndice(const std::vector<int> &shape, const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return ExpandTo1DIndice(shape, indices);
}

namespace {

class SubstituteMutator : ir::IRMutator<ir::Expr *> {
 public:
  SubstituteMutator(const std::map<const ir::_Var_ *, Expr> &var_map) {
    for (auto &item : var_map) {
      var_map_[item.first->name] = item.second;
    }
  }

  void operator()(ir::Expr *expr) { Visit(expr); }

 private:
  void Visit(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *op, ir::Expr *expr) override {
    auto it = var_map_.find(op->name);
    if (it == var_map_.end()) return;
    *expr = it->second;
  }

  Expr *expr_{};
  std::map<std::string, Expr> var_map_;
};

}  // namespace

void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map) {
  SubstituteMutator mutator(var_map);
  mutator(expr);
}

bool is_zero(Expr v) {
  auto *int_n   = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value = 0.f;
  return false;
}

}  // namespace common
}  // namespace cinn
