#include "cinn/common/ir_util.h"
#include <unordered_set>

#include "cinn/common/cas.h"
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

Expr IndiceToAbsOffset(const std::vector<Expr> &shape, const std::vector<Expr> &indices) {
  CHECK_GE(shape.size(), indices.size());
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

Expr IndiceToAbsOffset(const std::vector<int> &shape, const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return IndiceToAbsOffset(shape, indices);
}

Expr PrecedingAxisToAbsOffset(const std::vector<Expr> &shape, int preceding_n_axis) {
  std::vector<Expr> indices;
  for (int i = 0; i < preceding_n_axis; i++) indices.push_back(shape[i]);
  return IndiceToAbsOffset(shape, indices);
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
  v             = AutoSimplify(v);
  auto *int_n   = v.As<ir::IntImm>();
  auto *float_n = v.As<ir::FloatImm>();

  if (int_n) return int_n->value == 0;
  if (float_n) return float_n->value = 0.f;
  return false;
}

Expr CastIfNeeded(Expr body, Type type) {
  if (body.type() == type) return body;
  return ir::Cast::Make(type, body);
}

bool MathEqual(const Expr &a, const Expr &b) {
  auto c = a - b;
  c      = AutoSimplify(c);
  return is_zero(c);
}

Expr select(Expr cond, Expr true_value, Expr false_value) { return ir::Select::Make(cond, true_value, false_value); }

Expr and_all(const std::vector<Expr> &conds) {
  CHECK(!conds.empty());
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::And::Make(res, conds[i]);
  }
  return res;
}

Expr or_all(const std::vector<Expr> &conds) {
  CHECK(!conds.empty());
  Expr res = conds.front();
  for (int i = 1; i < conds.size(); i++) {
    res = ir::Or::Make(res, conds[i]);
  }
  return res;
}

void CheckTensorUniqueInExpr(Expr expr) {
  auto tensor_uniq = ir::CollectIRNodes(expr, [](const Expr *x) { return x->as_tensor(); });
  for (auto &t : tensor_uniq) LOG(INFO) << "found tensor: " << t << " " << t.as_tensor();
  std::unordered_map<std::string, const ir::_Tensor_ *> tensor_names;
  for (auto &t : tensor_uniq) {
    auto *tp = t.as_tensor();
    if (!tensor_names.count(tp->name)) {
      tensor_names[tp->name] = tp;
    } else {
      CHECK_EQ(tensor_names[tp->name], tp)
          << "Found tensor not unique [" << tp->name << "]\nThe original expression is \n"
          << expr;
    }
  }
}

void CheckBufferUniqueInExpr(Expr expr) {
  // the buffers exists in tensor and lowered functions.
  CheckTensorUniqueInExpr(expr);

  auto tensors = ir::CollectIRNodes(expr, [](const Expr *x) { return x->as_tensor(); });
  auto funcs   = ir::CollectIRNodes(expr, [](const Expr *x) { return x->as_lowered_func(); });

  std::unordered_map<std::string, const ir::_Buffer_ *> buffer_name;
  auto check_buffer_uniq = [&](const ir::_Buffer_ *b) {
    if (buffer_name.count(b->name)) {
      CHECK_EQ(buffer_name[b->name], b);
    } else {
      buffer_name[b->name] = b->const_self();
    }
  };
  for (auto &e : tensors) {
    auto *t = e.as_tensor();
    if (t->buffer.defined()) {
      check_buffer_uniq(t->buffer->const_self());
    }
  }

  for (auto &e : funcs) {
    auto *f = e.as_lowered_func();
    for (auto &b : f->temp_bufs) {
      if (b.defined()) {
        check_buffer_uniq(b->const_self());
      }
    }
  }
}

namespace {

struct AllTensorsUnifier : public ir::IRMutator<> {
  std::unordered_map<std::string, ir::_Tensor_ *> tensors;

  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Tensor_ *op, Expr *expr) override {
    auto *node = expr->as_tensor();
    if (tensors.count(node->name)) {
      if (tensors[node->name] != op) {
        expr->Reset(tensors[node->name]);
      }
    } else {
      tensors.emplace(node->name, node);
    }
  }
};

struct AllBuffersUnifier : public ir::IRMutator<> {
  std::unordered_map<std::string, ir::_Buffer_ *> buffers;

  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  // We assume the buffer only exists in Tensor and LoweredFunc.

  void UpdateBufferIfNeeded(ir::Buffer &buffer) {
    if (buffer.defined()) {
      auto &name = buffer->name;
      if (buffers.count(name)) {
        if (buffers[name] != buffer->const_self()) {
          buffer.Reset(buffers[name]);
        }
      } else {
        buffers[name] = buffer->self();
      }
    }
  }

  void Visit(const ir::_Tensor_ *op, Expr *expr) {
    auto *node = expr->As<ir::_Tensor_>();
    UpdateBufferIfNeeded(node->buffer);
  }

  void Visit(const ir::_LoweredFunc_ *op, Expr *expr) {
    auto *node = expr->As<ir::_LoweredFunc_>();
    for (auto &buffer : node->temp_bufs) {
      UpdateBufferIfNeeded(buffer);
    }
  }
};

}  // namespace

void UnifyAllTensorsInExpr(Expr *expr) { AllTensorsUnifier()(expr); }

void UnifyAllBuffersInExpr(Expr *expr) { AllBuffersUnifier()(expr); }

}  // namespace common
}  // namespace cinn
