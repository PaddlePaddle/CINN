#include <ginac/ginac.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace cinn {
namespace common {

TEST(GiNaC, simplify) {
  using namespace GiNaC;  // NOLINT
  symbol x("x");
  symbol y("y");

  ex e = x * 0 + 1 + 2 + 3 - 100 + 30 * y - y * 21 + 0 * x;
  LOG(INFO) << "e: " << e;
}

TEST(GiNaC, diff) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");
  ex e  = (x + 1);
  ex e1 = (y + 1);

  e  = diff(e, x);
  e1 = diff(e1, x);
  LOG(INFO) << "e: " << eval(e);
  LOG(INFO) << "e1: " << eval(e1);
}

TEST(GiNaC, solve) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");

  lst eqns{2 * x + 3 == 19};
  lst vars{x};

  LOG(INFO) << "solve: " << lsolve(eqns, vars);
  LOG(INFO) << diff(2 * x + 3, x);
}

}  // namespace common
}  // namespace cinn
