#include "cinn/utils/timer.h"

namespace cinn {
namespace utils {

float Timer::Stop() {
  end_     = std::chrono::system_clock::now();
  auto ts  = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
  float ms = 1000.f * static_cast<float>(ts.count()) * std::chrono::milliseconds::period::num /
             std::chrono::milliseconds::period::den;
  return ms;
}

void Timer::Start() { start_ = std::chrono::system_clock::now(); }

}  // namespace utils
}  // namespace cinn
