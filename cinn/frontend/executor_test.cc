#include "cinn/frontend/executor.h"

#include <gtest/gtest.h>
#include "cinn/runtime/use_extern_funcs.h"

DEFINE_string(model_dir, "", "");
DEFINE_string(resnet_model_dir, "", "");

namespace cinn::frontend {

TEST(Executor, basic) {
  Executor executor({"A"}, {{1, 30}});
  executor.LoadPaddleModel(FLAGS_model_dir);
  executor.Run();
}

TEST(Executor, resnet) {
  Executor executor({"resnet_input"}, {{1, 3, 224, 224}});
  executor.LoadPaddleModel(FLAGS_resnet_model_dir);
}

}  // namespace cinn::frontend
