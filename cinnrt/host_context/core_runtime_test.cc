#include "cinnrt/host_context/core_runtime.h"

#include <gtest/gtest.h>

#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/host_context/op_executable.h"
#include "cinnrt/host_context/symbol_table.h"

namespace cinnrt {
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
}  // namespace cinnrt
