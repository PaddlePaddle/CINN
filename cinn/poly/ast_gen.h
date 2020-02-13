/**
 * This file implements the isl AST build interface, it helps to generate isl AST given the polyhedral domain and
 * schedule.
 */
#pragma once
#include <isl/cpp.h>

#include <map>
#include <string>

#include "cinn/poly/element.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/schedule.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace poly {

class AstGen {
 public:
  AstGen(const isl::set& context, const std::vector<Element>& elements, const Scheduler& scheduler)
      : context_(context), poly_elements_(elements), scheduler_(scheduler) {}

  /**
   * Set forloop iterator names.
   * @param names
   * @return AstGen itself.
   */
  AstGen& SetIteratorNames(const std::vector<std::string>& names);

  isl::ctx ctx() const;

  isl::ast_node Build();

  const std::vector<Element>& poly_elements() const { return poly_elements_; }

  const std::map<std::string, isl::ast_expr>& axis2ast(const std::string& tuple_name) const;

 private:
  //! Return a domain composed of all the elements.
  isl::union_set domain();

  //! Return a map composed of all the transforms.
  isl::union_map transform();

  //! Replace the Expr with the transformed indices.
  //! e.g. Stage's expr is C[i,j] ...
  //! e.g. with ISL transformed statement S0(c0+1, c1*2), the expr will turn to C[c0+1, c1*2]
  static std::map<std::string, isl::ast_expr> ExtractIslTransformedIndiceMap(const isl::set& iterator_domain,
                                                                             isl_ast_build* build);

 private:
  isl::set context_;
  std::vector<Element> poly_elements_;
  const Scheduler& scheduler_;
  std::vector<std::string> iterator_names_;
  //! tuple name -> { axis -> isl_ast }
  std::map<std::string, std::map<std::string, isl::ast_expr>> transformed_indice_map_;
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
