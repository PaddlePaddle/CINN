#include "cinn/frontend/executor.h"
#include <gtest/gtest.h>

DEFINE_string(model_dir, "", "");

namespace cinn::frontend {

TEST(Executor, basic) {
  Executor executor({"A"}, {{1, 30}});
  executor.LoadPaddleModel(FLAGS_model_dir);
  executor.Run();
}

}  // namespace cinn::frontend
