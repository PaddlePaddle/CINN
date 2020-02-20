#include "cinn/common/target.h"

namespace cinn {
namespace common {

bool Target::operator==(const Target &other) const {
  return os == other.os &&      //
         arch == other.arch &&  //
         bits == other.bits &&  //
         features == other.features;
}

}  // namespace common
}  // namespace cinn
