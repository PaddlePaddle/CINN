#include "cinn/common/target.h"

#include <glog/logging.h>

#include "cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace common {

bool Target::operator==(const Target &other) const {
  return os == other.os &&      //
         arch == other.arch &&  //
         bits == other.bits &&  //
         features == other.features;
}

int Target::runtime_arch() const {
  switch (arch) {
    case Arch::Unk:
      return cinn_unk_device;
    case Arch::X86:
      return cinn_x86_device;
    case Arch::ARM:
      return cinn_arm_device;
    default:
      LOG(FATAL) << "Not supported arch";
  }
  return -1;
}

}  // namespace common
}  // namespace cinn
