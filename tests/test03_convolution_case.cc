#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/runtime/cinn_runtime.h"
#include "cinn/utils/timer.h"
#include "tests/test03_convolution_schedule1.h"

TEST(test03, schedule) {}

// include the generated C source code:
// @{
#include "tests/test03_convolution_schedule1.cc"
// @}
