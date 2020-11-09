#include "cinn/frontend/interpreter.h"

#include <gtest/gtest.h>

#include "cinn/runtime/use_extern_funcs.h"

DEFINE_string(model_dir, "", "");

namespace cinn::frontend {

TEST(Interpreter, basic) {
  Interpreter executor({"A"}, {{1, 30}});
  executor.LoadPaddleModel(FLAGS_model_dir, common::DefaultHostTarget());
  executor.Run();
  executor.GetTensor("fc_0.tmp_2");
}

}  // namespace cinn::frontend
