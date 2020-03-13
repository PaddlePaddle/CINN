/**
 * This file implements the isl AST build interface, it helps to generate isl AST given the polyhedral domain and
 * schedule.
 */
#pragma once
#include <isl/cpp.h>

#include <map>
#include <string>
#include <vector>

#include "cinn/lang/tensor.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/poly_scheduler.h"
#include "cinn/poly/schedule.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace poly {

/**
 * Generate IR from polyhedral schedule.
 */
class AstGen {
 public:
  AstGen(const isl::set& context, const std::vector<Stage*>& stages, const poly::ScheduleGroup& group);

  /**
   * Set forloop iterator names.
   * @param names
   * @return AstGen itself.
   */
  AstGen& SetIteratorNames(const std::vector<std::string>& names);

  isl::ctx ctx() const;

  isl::ast_node Build();

  //! Get the map from original CINN iterators to the transformed actual ISL ast nodes.
  const std::map<std::string, isl::ast_expr>& axis2ast(const std::string& tuple_name) const;

  void SetBuildOptions(const isl::union_map& options) { build_options_ = options; }

 private:
  //! Set the ISL ast_gen configs.
  void InitIslAstConfig();

  //! Return a domain composed of all the elements.
  isl::union_set domain();

  //! Return a map composed of all the transforms.
  isl::union_map transform();

  //! Replace the Expr with the transformed indices.
  //! e.g. Stage's expr is C[i,j] ...
  //! e.g. with ISL transformed statement S0(c0+1, c1*2), the expr will turn to C[c0+1, c1*2]
  static std::map<std::string, isl::ast_expr> ExtractIslTransformedIndiceMap(const isl::set& iterator_domain,
                                                                             isl_ast_build* build);

  //! Get the polyhedral stages.
  const std::vector<Shared<Stage>>& stages() const { return stages_; }

 private:
  isl::set context_;
  std::vector<Shared<Stage>> stages_;
  const poly::ScheduleGroup schedule_group_;
  std::vector<std::string> iterator_names_;
  //! tuple name -> { axis -> isl_ast }
  std::map<std::string, std::map<std::string, isl::ast_expr>> transformed_indice_map_;
  isl::union_map build_options_;
};

/**
 * Transform the isl ast to Expr.
 */
void IslAstNodeToCinnExpr(const isl::ast_node& node, ir::Expr* expr);
void IslAstExprToCinnExpr(const isl::ast_expr& node, ir::Expr* expr);

namespace detail {

//! Get tuple name of a ast node.
std::string GetTupleName(isl_ast_node* node);

}  // namespace detail

}  // namespace poly
}  // namespace cinn
