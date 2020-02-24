#include "cinn/poly/ast_gen.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace poly {

isl::union_set AstGen::domain() {
  CHECK(!stages_.empty());
  auto sets =
      utils::Map<std::vector<Shared<Stage>>, isl::set>(stages_, [](const Shared<Stage>& e) { return e->domain(); });
  return SetsToUnionSet(sets);
}

isl::ctx AstGen::ctx() const {
  CHECK(!stages_.empty());
  return stages_.front()->domain().ctx();
}

isl::ast_node AstGen::Build() {
  // Collect schedule from scheduler.
  auto schedules = scheduler_.BuildSchedule();
  std::vector<isl::map> maps;
  for (auto& stage : stages_) {
    auto it = schedules.find(stage->id());
    CHECK(it != std::end(schedules));
    maps.push_back(it->second);
  }
  auto schedule = MapsToUnionMap(maps);

  // Build it.
  auto ast_build = isl::ast_build::from_context(context_);
  // Set iterators.
  if (!iterator_names_.empty()) {
    auto iterator_names = scheduler_.WrapIteratorNames(iterator_names_);
    isl::id_list ids    = isl::manage(isl_id_list_alloc(ctx().get(), iterator_names.size()));
    for (int i = 0; i < iterator_names.size(); i++) {
      ids = isl::manage(isl_id_list_add(ids.release(), isl_id_alloc(ctx().get(), iterator_names[i].c_str(), nullptr)));
    }
    ast_build = isl::manage(isl_ast_build_set_iterators(ast_build.release(), ids.release()));
  }

  // collect iterator map
  auto get_domain_by_name = [this](const std::string& name) -> isl::set {
    auto ele_it =
        std::find_if(stages_.begin(), stages_.end(), [&name](const Shared<Stage>& ele) { return ele->id() == name; });
    CHECK(ele_it != std::end(stages_));
    return (*ele_it)->domain();
  };

  auto collect = [&](isl::ast_node node, isl::ast_build build) -> isl::ast_node {
    auto tuple_name                     = detail::GetTupleName(node.get());
    auto indice_map                     = ExtractIslTransformedIndiceMap(get_domain_by_name(tuple_name), build.get());
    transformed_indice_map_[tuple_name] = indice_map;
    return node;
  };

  ast_build = ast_build.set_at_each_domain(collect);

  isl::union_map transformed_schedule = transform().apply_range(schedule);
  auto schedule_domain                = transformed_schedule.intersect_domain(domain());
  VLOG(4) << "domain: " << domain();
  VLOG(4) << "transform schedule " << stages()[0]->transform();
  VLOG(4) << "schedule: " << schedule;
  VLOG(4) << "schedule_domain: " << schedule_domain;
  auto ast = ast_build.node_from_schedule_map(schedule_domain);
  VLOG(2) << "\n" << isl_ast_node_to_C_str(ast.get());
  return ast;
}

AstGen& AstGen::SetIteratorNames(const std::vector<std::string>& names) {
  iterator_names_ = names;
  return *this;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build* build, const isl::map& access);

std::map<std::string, isl::ast_expr> AstGen::ExtractIslTransformedIndiceMap(const isl::set& iterator_domain,
                                                                            isl_ast_build* build) {
  std::map<std::string, isl::ast_expr> iterator_map;
  isl::map identity = isl::manage(isl_set_identity(iterator_domain.copy()));
  isl::map schedule = identity;

  identity                = identity.apply_domain(schedule);
  isl::ast_expr idx_expr  = CreateIslAstIndexExpression(build, identity);
  isl::space domain_space = iterator_domain.space();

  for (int i = 1; i < isl_ast_expr_get_op_n_arg(idx_expr.get()); i++) {
    if (isl_space_has_dim_name(domain_space.get(), isl_dim_set, i - 1)) {
      std::string original_idx_name   = isl_space_get_dim_name(domain_space.get(), isl_dim_set, i - 1);
      isl::ast_expr transformed_index = isl::manage(isl_ast_expr_get_op_arg(idx_expr.get(), i));
      iterator_map.emplace(original_idx_name, transformed_index);
    }
  }

  return iterator_map;
}

const std::map<std::string, isl::ast_expr>& AstGen::axis2ast(const std::string& tuple_name) const {
  auto it = transformed_indice_map_.find(tuple_name);
  CHECK(it != transformed_indice_map_.end()) << "no id " << tuple_name;
  return it->second;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build* build, const isl::map& access) {
  CHECK(build);
  isl::map schedule = isl::manage(isl_map_from_union_map(isl_ast_build_get_schedule(build)));

  // get identity access from schedule.
  auto statement       = isl_map_get_statement_repr(schedule.get(), isl_dim_in);
  auto statement_set   = isl::manage(isl_set_read_from_str(isl_map_get_ctx(schedule.get()),
                                                         utils::StringFormat("{ %s : }", statement.c_str()).c_str()));
  auto identity_access = isl::manage(isl_set_identity(statement_set.release()));
  isl::map map         = isl::manage(isl_map_reverse(schedule.copy()));

  isl::pw_multi_aff iterator_map = isl::manage(isl_pw_multi_aff_from_map(map.copy()));
  isl::pw_multi_aff index_aff    = isl::manage(isl_pw_multi_aff_from_map(identity_access.copy()));

  isl::space model2        = iterator_map.space();
  index_aff                = isl::manage(isl_pw_multi_aff_align_params(index_aff.copy(), model2.copy()));
  isl::space model         = index_aff.space();
  iterator_map             = isl::manage(isl_pw_multi_aff_align_params(iterator_map.copy(), model.copy()));
  iterator_map             = isl::manage(isl_pw_multi_aff_pullback_pw_multi_aff(index_aff.copy(), iterator_map.copy()));
  isl::ast_expr index_expr = isl::manage(isl_ast_build_access_from_pw_multi_aff(build, iterator_map.copy()));

  return index_expr;
}

isl::union_map AstGen::transform() {
  std::vector<isl::map> transforms;
  for (auto& stage : stages()) {
    transforms.push_back(stage->transform());
  }
  return MapsToUnionMap(transforms);
}

namespace detail {

std::string GetTupleName(isl_ast_node* node) {
  auto expr = isl::manage(isl_ast_node_user_get_expr(node));
  auto arg  = isl::manage(isl_ast_expr_get_op_arg(expr.get(), 0));
  auto name = isl_id_get_name(isl_ast_expr_get_id(arg.get()));
  return name;
}

}  // namespace detail

//! Eat an isl block node.
void EatBlock(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl user node.
void EatUser(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl for node.
void EatFor(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl `if` node.
void EatIf(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl mark node.
void EatMark(const isl::ast_node& node, ir::Expr* expr);

void IslAstNodeToCinnExpr(const isl::ast_node& node, ir::Expr* expr) {
  CHECK(!node.is_null());
  CHECK(expr);

  switch (isl_ast_node_get_type(node.get())) {
    case isl_ast_node_block: {
      VLOG(4) << "get isl block node";
      EatBlock(node, expr);
    } break;
    case isl_ast_node_for: {
      VLOG(4) << "get isl for node";
      EatFor(node, expr);
    } break;
    case isl_ast_node_if: {
      VLOG(4) << "get isl if node";
      EatIf(node, expr);
    } break;
    case isl_ast_node_user: {
      VLOG(4) << "get isl user node";
      EatUser(node, expr);
    } break;
    case isl_ast_node_mark: {
      VLOG(4) << "get isl mark";
      // EatMark(node, expr);
    } break;
    default:
      LOG(FATAL) << "Unexpected ISL node type " << isl_ast_node_get_type(node.get());
      break;
  }
}

// Eat an isl block node.
void EatBlock(const isl::ast_node& node, ir::Expr* expr) {
  VLOG(2) << "get isl ast body node";
  CHECK(!node.is_null());
  CHECK(expr);
  CHECK_EQ(isl_ast_node_get_type(node.get()), isl_ast_node_block);
  isl::ast_node_list list = isl::manage(isl_ast_node_block_get_children(node.get()));
  std::vector<ir::Expr> exprs;
  for (int i = 0; i < isl_ast_node_list_n_ast_node(list.get()); i++) {
    isl::ast_node child = isl::manage(isl_ast_node_list_get_ast_node(list.get(), i));
    // visit child
    ir::Expr child_expr;
    IslAstNodeToCinnExpr(child, &child_expr);
    exprs.push_back(child_expr);
  }
  *expr = ir::Block::Make(std::move(exprs));
}
// Eat an isl user node.
void EatUser(const isl::ast_node& node, ir::Expr* expr) {
  CHECK_EQ(isl_ast_node_get_type(node.get()), isl_ast_node_user);
  isl::ast_expr isl_expr = isl::manage(isl_ast_node_user_get_expr(node.get()));
  IslAstExprToCinnExpr(isl_expr, expr);
}
// Eat an isl `for` node.
void EatFor(const isl::ast_node& node, ir::Expr* expr) {
  CHECK_EQ(isl_ast_node_get_type(node.get()), isl_ast_node_for);

  // iter name
  isl::ast_expr iter    = isl::manage(isl_ast_node_for_get_iterator(node.get()));
  isl::id iter_id       = isl::manage(isl_ast_expr_get_id(iter.get()));
  std::string iter_name = iter_id.name();

  // get condition
  isl::ast_expr condition   = isl::manage(isl_ast_node_for_get_cond(node.get()));
  isl::ast_expr incrementor = isl::manage(isl_ast_node_for_get_inc(node.get()));
  isl::ast_expr initializer = isl::manage(isl_ast_node_for_get_init(node.get()));
  isl::ast_node body        = isl::manage(isl_ast_node_for_get_body(node.get()));

  ir::Expr ir_body;
  IslAstNodeToCinnExpr(body, &ir_body);
  ir_body = ir::Block::Make({ir_body});

  ir::Expr ir_initializer;
  IslAstExprToCinnExpr(initializer, &ir_initializer);

  ir::Expr ir_condition;
  IslAstExprToCinnExpr(condition, &ir_condition);
  ir::Expr tmp;

  isl::ast_expr arg = isl::manage(isl_ast_expr_get_op_arg(condition.get(), 1));
  IslAstExprToCinnExpr(arg, &tmp);

  ir::Expr ir_inc;
  IslAstExprToCinnExpr(incrementor, &ir_inc);

  ir::Var ir_iter(iter_name);

  *expr = ir::PolyFor::Make(
      ir::Var(iter_name), ir_initializer, ir_condition, ir_inc, ir::ForType::Serial, ir::DeviceAPI ::Host, ir_body);
}

void EatIf(const isl::ast_node& node, ir::Expr* expr) {
  CHECK_EQ(isl_ast_node_get_type(node.get()), isl_ast_node_if);
  isl::ast_node then_body = isl::manage(isl_ast_node_if_get_then(node.get()));
  isl::ast_expr condition = isl::manage(isl_ast_node_if_get_cond(node.get()));

  ir::Expr ir_then_body;
  IslAstNodeToCinnExpr(then_body, &ir_then_body);

  ir::Expr ir_else_body;
  if (isl_bool_true == isl_ast_node_if_has_else(node.get())) {
    isl::ast_node else_body = isl::manage(isl_ast_node_if_get_else(node.get()));
    IslAstNodeToCinnExpr(else_body, &ir_else_body);
  }

  ir::Expr ir_condition;
  IslAstExprToCinnExpr(condition, &ir_condition);

  if (ir_else_body.defined()) {
    *expr = ir::IfThenElse::Make(ir_condition, ir_then_body, ir_else_body);
  } else {
    *expr = ir::IfThenElse::Make(ir_condition, ir_then_body, ir::Expr());
  }
}

void IslAstExprToCinnExpr(const isl::ast_expr& node, ir::Expr* expr) {
  switch (isl_ast_expr_get_type(node.get())) {
    case isl_ast_expr_int: {
      isl::val val = isl::manage(isl_ast_expr_get_val(node.get()));
      *expr        = ir::Expr(static_cast<int>(isl_val_get_num_si(val.get())));
    } break;
    case isl_ast_expr_id: {
      isl::id id = isl::manage(isl_ast_expr_get_id(node.get()));
      *expr      = ir::Var(id.name());
    } break;
    case isl_ast_expr_op: {
      std::vector<ir::Expr> ops;
      const int n_args = isl_ast_expr_get_op_n_arg(node.get());

      for (int i = 0; i < n_args; i++) {
        ir::Expr op;
        isl::ast_expr expr0 = isl::manage(isl_ast_expr_get_op_arg(node.get(), i));
        IslAstExprToCinnExpr(expr0, &op);
        ops.push_back(op);
      }

      auto set_ops_ptype = [&](ir::Type type) {
        for (auto& op : ops) {
          op->set_type(type);
        }
      };

      // set ops as int32 by default.
      set_ops_ptype(Int(32));

      isl_ast_op_type op_type = isl_ast_expr_get_op_type(node.get());
      switch (op_type) {
        case isl_ast_op_and: {
          set_ops_ptype(Bool());
          *expr = ir::And::Make(ops[0], ops[1]);
        } break;
        case isl_ast_op_or:
          *expr = ir::Or::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_min:
          *expr = ir::Min::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_max:
          *expr = ir::Max::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_minus:
          *expr = ir::Minus::Make(ops[0]);
          break;
        case isl_ast_op_add:
          *expr = ir::Add::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_sub:
          *expr = ir::Sub::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_mul:
          *expr = ir::Mul::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_div:
          *expr = ir::Div::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_le:
          *expr = ir::LE::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_lt:
          *expr = ir::LT::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_ge:
          *expr = ir::GE::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_gt:
          *expr = ir::GT::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_eq:
          *expr = ir::EQ::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_call: {
          ir::Expr caller_expr = ops.front();
          // TODO(Superjomn) make it an string
          CHECK(caller_expr.node_type() == ir::IrNodeTy::_Var_);
          std::string caller = caller_expr.As<ir::_Var_>()->name;
          ops.erase(ops.begin());
          // NOTE the type here is not important.
          *expr = ir::Call::Make(Float(32), caller, ops, ir::Call::ISL);
        } break;
        case isl_ast_op_fdiv_q:
          *expr = ir::Div::Make(ops[0], ops[1]);
          break;
        default:
          LOG(FATAL) << "unsupported op " << op_type;
      }
    } break;
    default:
      break;
  }
}

void AstGen::InitIslAstConfig() {
  isl_options_set_ast_build_detect_min_max(ctx().get(), 1);
  isl_options_set_ast_build_exploit_nested_bounds(ctx().get(), 1);
  isl_options_set_ast_build_scale_strides(ctx().get(), 1);
  isl_options_set_ast_build_allow_else(ctx().get(), 1);
}

AstGen::AstGen(const isl::set& context, const std::vector<Stage*>& stages, const Scheduler& scheduler)
    : context_(context), scheduler_(scheduler) {
  for (auto* x : stages) stages_.emplace_back(x);
  InitIslAstConfig();
}

}  // namespace poly
}  // namespace cinn
