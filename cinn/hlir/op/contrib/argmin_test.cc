#include "cinn/hlir/op/contrib/argmin.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/common/context.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {

TEST(GenerateCode_Cpu, Argmin_Keep) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  int axis = 1;
  ir::Expr n(4);
  ir::Expr in_c(3);
  ir::Expr out_c(1);
  ir::Expr h(28);
  ir::Expr w(28);

  lang::Placeholder<float> in("in", {n, in_c, h, w});
  lang::Placeholder<float> out("out", {n, out_c, h, w});
  ir::Tensor res = Argmin(in, axis, true, "test_argmin_in");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Argmin_Keep", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Argmin_Keep_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;
}

TEST(GenerateCode_Cpu, Argmin) {
  common::Context::Global().ResetNameId();

  common::Target target = common::DefaultHostTarget();

  int axis = 1;
  ir::Expr n(4);
  ir::Expr in_c(3);
  ir::Expr h(28);
  ir::Expr w(28);

  lang::Placeholder<float> in("in", {n, in_c, h, w});
  lang::Placeholder<int> out("out", {n, h, w});
  ir::Tensor res = Argmin(in, axis, true, "test_argmin_in");

  poly::StageMap stages = poly::CreateStages({res});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestGenerateCodeCpu_Argmin", stages, {res}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr before CPU codegen:";
  VLOG(6) << funcs[0]->body;

  ir::Module::Builder builder("Argmin_Module", target);
  for (auto& f : funcs) {
    builder.AddFunction(f);
  }

  backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
  VLOG(6) << "Cpu Codegen result:";
  VLOG(6) << code << std::endl;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn
