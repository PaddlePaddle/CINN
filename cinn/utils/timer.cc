#include "cinn/utils/timer.h"

namespace cinn {
namespace utils {

float Timer::Stop() {
  end_     = std::chrono::high_resolution_clock::now();
  auto ts  = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_);
  float ms = 1000. * static_cast<double>(ts.count()) * std::chrono::nanoseconds::period::num /
             std::chrono::nanoseconds::period::den;
  return ms;
}

void Timer::Start() { start_ = std::chrono::high_resolution_clock::now(); }

}  // namespace utils
}  // namespace cinn
