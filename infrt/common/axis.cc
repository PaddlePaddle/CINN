#include "infrt/common/axis.h"

#include <glog/logging.h>

#include <set>

#include "infrt/common/common.h"

namespace infrt {
namespace common {

const std::vector<std::string> kAxises({
    "i",  // level 0
    "j",  // level 1
    "k",  // level 2
    "a",  // level 3
    "b",  // level 4
    "c",  // level 5
    "d",  // level 6
    "e",  // level 7
    "f",  // level 8
    "g",  // level 9
    "h"   // level 10
});

static std::set<std::string> axis_set() {
  static std::set<std::string> x(kAxises.begin(), kAxises.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) { return axis_set().count(x); }

const std::string& axis_name(int level) {
  CHECK_LT(level, kAxises.size());
  return kAxises[level];
}

}  // namespace common
}  // namespace infrt
