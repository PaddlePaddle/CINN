#include "cinn/utils/string.h"

#include <gtest/gtest.h>

namespace cinn {
namespace utils {

TEST(string, Endswith) {
  std::string a = "a__p";
  ASSERT_TRUE(Endswith(a, "__p"));
  ASSERT_FALSE(Endswith(a, "_x"));
  ASSERT_TRUE(Endswith(a, "a__p"));
  ASSERT_FALSE(Endswith(a, "a___p"));
}
TEST(string, Startswith) {
  std::string a = "a__p";
  ASSERT_TRUE(Startswith(a, "a_"));
  ASSERT_TRUE(Startswith(a, "a__"));
  ASSERT_TRUE(Startswith(a, "a__p"));
  ASSERT_FALSE(Startswith(a, "a___p"));
}

}  // namespace utils
}  // namespace cinn
