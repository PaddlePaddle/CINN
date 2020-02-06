#include "cinn/poly/ast_gen.h"

namespace cinn {
namespace poly {

isl::ast_node AstGen::operator()(const std::vector<Element> &elements, const Scheduler &scheduler) {
  // Collect domains.
  auto sets = utils::Map<std::vector<Element>, isl::set>(elements, [](const Element &e) { return e.domain(); });
  isl::union_set domain = SetsToUnionSet(sets);

  isl::ctx ctx = elements.front().domain().ctx();

  // Collect schedule from scheduler.
  auto schedules = scheduler.BuildSchedule();
  std::vector<isl::map> maps;
  for (auto &ele : elements) {
    auto it = schedules.find(ele.id());
    CHECK(it != std::end(schedules));
    maps.push_back(it->second);
  }
  auto schedule = MapsToUnionMap(maps);

  // Build it.
  auto build = isl::ast_build::from_context(context_);
  // Set iterators.
  if (!iterator_names_.empty()) {
    auto iterator_names = scheduler.WrapIteratorNames(iterator_names_);
    isl::id_list ids    = isl::manage(isl_id_list_alloc(ctx.get(), iterator_names.size()));
    for (int i = 0; i < iterator_names.size(); i++) {
      ids = isl::manage(isl_id_list_add(ids.release(), isl_id_alloc(ctx.get(), iterator_names[i].c_str(), nullptr)));
    }
    build = isl::manage(isl_ast_build_set_iterators(build.release(), ids.release()));
  }

  auto ast = build.node_from_schedule_map(schedule.intersect_domain(domain));
  VLOG(2) << "\n" << isl_ast_node_to_C_str(ast.get());
  return ast;
}

AstGen &AstGen::SetIteratorNames(const std::vector<std::string> &names) {
  iterator_names_ = names;
  return *this;
}

}  // namespace poly
}  // namespace cinn
