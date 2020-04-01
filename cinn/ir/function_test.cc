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

TEST(Function, test1) {
  PackedFunc::func_t body = [](Args args, RetValue* ret) {
    auto* msg = static_cast<const char*>(args[0]);

    ret->Set(msg);
  };

  PackedFunc func(body);
  const char* msg = "hello world";
  char* c         = func(msg);
  LOG(INFO) << static_cast<char*>(c);
}

}  // namespace ir
}  // namespace cinn