#include "infrt/host_context/op_executable.h"

#include <gtest/gtest.h>

#include "infrt/host_context/kernel_registry.h"
#include "infrt/host_context/kernel_utils.h"
#include "infrt/host_context/symbol_table.h"

namespace infrt {
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
  ASSERT_EQ(results.front()->get<int32_t>(), 3);

  // check symbol table contains the same result instance.
  LOG(INFO) << "type: " << table.GetValue("c")->type_info();
  int c = table.GetValue("c")->get<int32_t>();
  ASSERT_EQ(c, 3);
}

}  // namespace host_context
}  // namespace infrt
