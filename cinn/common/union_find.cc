#include "cinn/common/union_find.h"

namespace cinn {
namespace common {

const char* UnionFindNode::__type_info__ = "UnionFindNode";
const char* UnionFindNode::type_info() const { return __type_info__; }

}  // namespace common
}  // namespace cinn
