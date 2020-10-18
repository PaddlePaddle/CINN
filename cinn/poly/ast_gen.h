/**
 * This file implements the isl AST build interface, it helps to generate isl AST given the polyhedral domain and
 * schedule.
 */
#pragma once
#include <isl/cpp.h>

#include <map>
#include <string>
#include <vector>

#include "cinn/ir/tensor.h"
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
  ~AstGen();

  /**
   * Set for-loop iterator names.
   * @param names
   * @return AstGen itself.
   */
  AstGen& SetIteratorNames(const std::vector<std::string>& names);

  isl::ctx ctx() const;

  isl::ast_node Build();

  //! Get the map from original CINN iterators to the transformed actual ISL ast nodes.
  const std::map<std::string, isl::ast_expr>& axis2ast(const std::string& tuple_name) const;

  const std::map<std::string, Expr> axis2expr(const std::string& tuple_name) const;

  bool ContainsStatement(const std::string& name) const;

  void SetBuildOptions(const isl::union_map& options);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
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
