#include "cinn/hlir/instruction/lower_impl.h"

namespace cinn {
namespace hlir {
namespace instruction {

LowerImplRegistry& LowerImplRegistry::Global() {
  static LowerImplRegistry x;
  return x;
}

void LowerImplRegistry::Insert(InstrCode code, const std::string& kind, std::function<LowerImplBase*()>&& creator) {
  CHECK(!data_.count(code) || !data_[code].count(kind))
      << "duplicate insert LowerImpl called [" << code << ":" << kind << "]";
  data_[code].emplace(kind, std::move(creator));
}

LowerImplBase* LowerImplRegistry::Create(InstrCode code, const std::string& kind) {
  auto it = data_.find(code);
  if (it == data_.end()) return nullptr;
  auto it1 = it->second.find(kind);
  if (it1 == it->second.end()) return nullptr;
  return it1->second();
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn