#pragma once
#include <algorithm>
#include <chrono>  //NOLINT
#include <vector>

namespace cinn {
namespace utils {

class Timer {
 public:
  Timer() = default;

  void Start();
  float Stop();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start_, end_;
};

}  // namespace utils
}  // namespace cinn
