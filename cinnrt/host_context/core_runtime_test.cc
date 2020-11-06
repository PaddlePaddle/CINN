#include "core_runtime.h"

#include <gtest/gtest.h>

#include "kernel_registry.h"
#include "kernel_utils.h"
#include "op_executable.h"
#include "symbol_table.h"

namespace cinn {
namespace host_context {

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

TEST(CoreRuntime, basic) {
  KernelRegistry registry;
  registry.AddKernel("cinn.test.addi32", CINN_KERNEL(add));
  registry.AddKernel("cinn.test.subi32", CINN_KERNEL(sub));

  CoreRuntimeBuilder builder(&registry);
  auto* table = builder.NewSymbolTable("main");
  table->Register("a", 1);
  table->Register("b", 2);
  table->Register("d", 4);

  // c = a + b
  auto* op0 = builder.NewOpExecutable("cinn.test.addi32", "main");
  op0->AppendArgument("a");
  op0->AppendArgument("b");
  op0->SetResults({"c"});

  // e = c - d
  auto* op1 = builder.NewOpExecutable("cinn.test.subi32", "main");
  op1->AppendArgument("c");
  op1->AppendArgument("d");
  op1->SetResults({"e"});

  builder.Execute();

  ASSERT_EQ(table->Get("d")->get<int>(), 4);
  ASSERT_EQ(table->Get("c")->get<int>(), 3);

  ASSERT_EQ(table->Get("e")->get<int>(), -1);
}

}  // namespace host_context
}  // namespace cinn
