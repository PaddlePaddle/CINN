#include "cinn/ir/function.h"
#include <gtest/gtest.h>

namespace cinn {
namespace ir {

TEST(Function, test) {
  PackedFunc::func_t func_body = [](Args args, RetValue* ret) {
    int a = args[0];
    int b = args[1];
    ret->Set(a + b);
  };
  PackedFunc func(func_body);

  int c = func(1, 2);
  LOG(INFO) << "c " << c;
}

}  // namespace ir
}  // namespace cinn