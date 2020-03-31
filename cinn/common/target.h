#pragma once

#include <vector>

namespace cinn {
namespace common {

struct Target {
  /**
   * The operating system used by the target. Determines which system calls to generate.
   */
  enum class OS : int {
    Unk = -1,
    Linux,
    Windows,
  };

  /**
   * The architecture used by the target. Determines the instruction set to use.
   */
  enum class Arch : int {
    Unk = -1,
    X86,
    ARM,
  };

  enum Bit : int {
    Unk = -1,
    k32,
    k64,
  };

  OS os{OS::Unk};
  Arch arch{Arch::Unk};
  Bit bits{Unk};

  enum class Feature : int {
    JIT = 0,
    Debug,
  };
  std::vector<Feature> features;

  Target(OS o = OS::Linux, Arch a = Arch::Unk, Bit b = Bit::Unk, const std::vector<Feature>& features = {})
      : os(o), arch(a), bits(b), features(features) {}

  bool defined() const { return os != OS::Unk && arch != Arch::Unk && bits != Bit::Unk; }

  //! Get the Runtime architecture, it is casted to integer to avoid header file depending.
  int runtime_arch() const;

  bool operator==(const Target& other) const;
  bool operator!=(const Target& other) const { return !(*this == other); }
};

static const Target& UnkTarget() {
  static Target target(Target::OS::Unk, Target::Arch::Unk, Target::Bit::Unk, {});
  return target;
}

}  // namespace common
}  // namespace cinn
