#include "cinn/common/target.h"

#include <glog/logging.h>

#include <sstream>

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

int Target::max_num_threads() const {
  CHECK(arch == Arch::NVGPU) << "The target is not NVGPU! Cannot get max number of threads.";
  return 1024;
}

std::vector<Target::Lib> Target::get_target_libs() const { return libs; }

int Target::get_target_bits() const {
  switch (bits) {
    case Bit::k32:
      return 32;
    case Bit::k64:
      return 64;
    case Bit::Unk:
      return 0;
    default:
      LOG(FATAL) << "Not supported Bit";
  }
  return -1;
}

std::string Target::arch_str() const {
  std::ostringstream oss;
  oss << arch;
  return oss.str();
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
    case Target::Arch::NVGPU:
      os << "nvgpu";
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

std::ostream &operator<<(std::ostream &os, Target::Arch arch) {
  switch (arch) {
    case Target::Arch::Unk:
      os << "Unk";
      break;
    case Target::Arch::X86:
      os << "X86";
      break;
    case Target::Arch::ARM:
      os << "ARM";
      break;
    case Target::Arch::NVGPU:
      os << "NVGPU";
      break;
  }
  return os;
}

}  // namespace common
}  // namespace cinn
