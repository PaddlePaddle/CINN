#include "cinn/poly/map.h"

namespace cinn {
namespace poly {

std::string Map::__str__() const {
  CHECK(!domain_iterators_.empty());

  auto get_ids_repr = [](const std::vector<Iterator>& ids) {
    std::vector<std::string> fields;
    std::transform(ids.begin(), ids.end(), std::back_inserter(fields), [](const Iterator& x) { return x.id; });
    return utils::Join(fields, ", ");
  };

  auto domain_iterators_repr = get_ids_repr(domain_iterators_);
  auto range_iterators_repr  = get_ids_repr(range_iterators_);

  std::vector<std::string> conds_fields;
  std::transform(
      conds_.begin(), conds_.end(), std::back_inserter(conds_fields), [](const Condition& x) { return x.__str__(); });
  auto conds_repr = utils::Join(conds_fields, " and ");

  return utils::StringFormat("{ %s[%s] -> %s[%s]: %s }",
                             id_.c_str(),
                             domain_iterators_repr.c_str(),
                             range_id_.c_str(),
                             range_iterators_repr.c_str(),
                             conds_repr.c_str());
}

Map::Map(isl::ctx ctx,
         std::string id,
         std::vector<Iterator> domain_iterators,
         std::vector<Iterator> range_iterators,
         std::vector<Condition> conds,
         std::string range_id)
    : ctx_(ctx),
      id_(std::move(id)),
      domain_iterators_(std::move(domain_iterators)),
      range_iterators_(std::move(range_iterators)),
      conds_(std::move(conds)),
      range_id_(std::move(range_id)) {}

isl::map Map::to_isl() const { return isl::map(ctx_, __str__()); }

std::ostream& operator<<(std::ostream& os, const Iterator& x) {
  os << utils::StringFormat("<Iterator: %s>", x.id.c_str());
  return os;
}

}  // namespace poly
}  // namespace cinn
