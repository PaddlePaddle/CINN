#pragma once

#include <vector>

namespace cinn {
namespace utils {

struct Target {
  /**
   * The operating system used by the target. Determines which system calls to generate.
   */
  enum class OS : int {
    Unk = -1,
    Linux,
    Windows,
  };
  OS os{Unk};

  /**
   * The architecture used by the target. Determines the instruction set to use.
   */
  enum class Arch : int {
    Unk = -1,
    X86,
    ARM,
  };
  Arch arch{Unk};

  enum Bit : int {
    Unk = -1,
    k32,
    k64,
  };
  Bit bits{Unk};

  enum class Feature : int {
    JIT = 0,
    Debug,
  };
  std::vector<Feature> features;

  Target() = default;

  Target(OS o, Arch a, Bit b, const std::vector<Feature>& features) : os(o), arch(a), bits(b) {}

  bool operator==(const Target& other) const;
  bool operator!=(const Target& other) const { return !(*this == other); }
};
}  // namespace utils
}  // namespace cinn
