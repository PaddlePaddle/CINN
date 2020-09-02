#include "cinn/frontend/paddle/model_parser.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_string(model_dir, "<NOTEXIST>", "model directory path");

namespace cinn::frontend::paddle {

TEST(LoadModelPb, naive_model) {
  hlir::framework::Scope scope;
  cpp::ProgramDesc program_desc;
  LoadModelPb(FLAGS_model_dir, "__model__", "", &scope, &program_desc, false);

  ASSERT_EQ(program_desc.BlocksSize(), 1UL);

  auto* block = program_desc.GetBlock<cpp::BlockDesc>(0);
  ASSERT_EQ(block->OpsSize(), 4UL);
  for (int i = 0; i < block->OpsSize(); i++) {
    auto* op = block->GetOp<cpp::OpDesc>(i);
    LOG(INFO) << op->Type();
  }

  // The Op list:
  // feed
  // mul
  // scale
  // fetch
}

}  // namespace cinn::frontend::paddle
