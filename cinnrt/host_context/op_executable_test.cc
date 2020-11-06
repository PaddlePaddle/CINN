#include "cinnrt/host_context/op_executable.h"

#include <gtest/gtest.h>

#include "cinnrt/host_context/kernel_registry.h"
#include "cinnrt/host_context/kernel_utils.h"
#include "cinnrt/host_context/symbol_table.h"

namespace cinn {
namespace host_context {

int add(int a, int b) { return a + b; }

TEST(OpExecutable, basic) {
  // register kernel
  KernelRegistry registry;
  registry.AddKernel("cinn.test.add.i32", CINN_KERNEL(add));

  SymbolTable table;
  table.Register("a", 1);
  table.Register("b", 2);

  OpExecutableBuilder executable("cinn.test.add.i32", &table, &registry);
  executable.AppendArgument("a");
  executable.AppendArgument("b");
  executable.SetResults({"c"});

  executable.Execute();

  // check the kernel frame has the result.
  auto results = executable.frame().GetResults();
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().get<int>(), 3);

  // check symbol table contains the same result instance.
  int c = table.Get("c")->get<int>();
  ASSERT_EQ(c, 3);
}

}  // namespace host_context
}  // namespace cinn
