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

std::ostream &operator<<(std::ostream &os, const Target &target) {
  os << "Target<";
  switch (target.os) {
    case Target::OS::Linux:
      os << "linux";
      break;
    case Target::OS::Windows:
      os << "windows";
      break;
    case Target::OS::Unk:
      os << "unk";
      break;
  }

  os << ",";

  switch (target.arch) {
    case Target::Arch::X86:
      os << "x86";
      break;
    case Target::Arch::ARM:
      os << "arm";
      break;
    case Target::Arch::Unk:
      os << "unk";
      break;
  }
  os << ",";

  switch (target.bits) {
    case Target::Bit::k32:
      os << "32";
      break;
    case Target::Bit::k64:
      os << "64";
      break;
    case Target::Bit::Unk:
      os << "unk";
      break;
  }
  os << ">";

  return os;
}

}  // namespace common
}  // namespace cinn
