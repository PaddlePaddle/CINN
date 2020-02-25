#pragma once

#include <isl/cpp.h>

#include <string>
#include <vector>

#include "cinn/poly/dim.h"

namespace cinn {
namespace poly {

struct Domain {
  //! The id of the statement.
  std::string id;
  //! The dimensions.
  std::vector<Dim> dims;

  //! The ISL context.
  isl::ctx ctx;

  Domain(isl::ctx ctx, std::string id, std::vector<Dim> dims) : ctx(ctx), id(std::move(id)), dims(std::move(dims)) {}

  //! The ISL format representation, such as '{ S[i]: 0<=i<=20 }'.
  std::string __str__() const;

  //! Get the isl domain.
  isl::set to_isl() const;
};

}  // namespace poly
}  // namespace cinn
