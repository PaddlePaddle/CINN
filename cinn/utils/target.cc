#include "cinn/utils/target.h"

namespace cinn {
namespace utils {

bool Target::operator==(const Target &other) const {
  return os == other.os &&      //
         arch == other.arch &&  //
         bits == other.bits &&  //
         features == other.features;
}

}  // namespace utils
}  // namespace cinn
