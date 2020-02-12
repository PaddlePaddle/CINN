#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace poly {

isl::union_set AstGen::domain() {
  CHECK(!poly_elements_.empty());
  auto sets = utils::Map<std::vector<Element>, isl::set>(poly_elements_, [](const Element &e) { return e.domain(); });
  return SetsToUnionSet(sets);
}

isl::ctx AstGen::ctx() const {
  CHECK(!poly_elements_.empty());
  return poly_elements_.front().domain().ctx();
}

isl::ast_node AstGen::Build() {
  // Collect schedule from scheduler.
  auto schedules = scheduler_.BuildSchedule();
  std::vector<isl::map> maps;
  for (auto &ele : poly_elements_) {
    auto it = schedules.find(ele.id());
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
  auto get_domain_by_name = [this](const std::string &name) -> isl::set {
    auto ele_it = std::find_if(
        poly_elements_.begin(), poly_elements_.end(), [&name](const Element &ele) { return ele.id() == name; });
    CHECK(ele_it != std::end(poly_elements_));
    return ele_it->domain();
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
  VLOG(4) << "transform schedule " << poly_elements()[0].schedule();
  VLOG(4) << "schedule: " << schedule;
  VLOG(4) << "schedule_domain: " << schedule_domain;
  auto ast = ast_build.node_from_schedule_map(schedule_domain);
  VLOG(2) << "\n" << isl_ast_node_to_C_str(ast.get());
  return ast;
}

AstGen &AstGen::SetIteratorNames(const std::vector<std::string> &names) {
  iterator_names_ = names;
  return *this;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build *build, const isl::map &access);

std::map<std::string, isl::ast_expr> AstGen::ExtractIslTransformedIndiceMap(const isl::set &iterator_domain,
                                                                            isl_ast_build *build) {
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

const std::map<std::string, isl::ast_expr> &AstGen::axis2ast(const std::string &tuple_name) const {
  auto it = transformed_indice_map_.find(tuple_name);
  CHECK(it != transformed_indice_map_.end()) << "no id " << tuple_name;
  return it->second;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build *build, const isl::map &access) {
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
  for (auto &ele : poly_elements()) {
    transforms.push_back(ele.schedule());
  }
  return MapsToUnionMap(transforms);
}

namespace detail {

std::string GetTupleName(isl_ast_node *node) {
  auto expr = isl::manage(isl_ast_node_user_get_expr(node));
  auto arg  = isl::manage(isl_ast_expr_get_op_arg(expr.get(), 0));
  auto name = isl_id_get_name(isl_ast_expr_get_id(arg.get()));
  return name;
}

}  // namespace detail

}  // namespace poly
}  // namespace cinn
