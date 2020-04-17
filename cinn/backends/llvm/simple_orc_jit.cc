#include "cinn/backends/llvm/simple_orc_jit.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <utility>

#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

SimpleOrcJit::SimpleOrcJit(llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout data_layout)
    : object_layer_(execution_session_, std::bind(std::make_unique<llvm::SectionMemoryManager>)),
      compile_layer_(
          execution_session_, object_layer_, std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb))),
      data_layout_(std::move(data_layout)),
      mangle_(execution_session_, data_layout_),
      context_(std::make_unique<llvm::LLVMContext>()),
      main_jd_(execution_session_.createJITDylib("<main>")) {
  main_jd_.addGenerator(
      llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(data_layout_.getGlobalPrefix())));
}

/*static*/ std::unique_ptr<SimpleOrcJit> SimpleOrcJit::Create() {
  LLVMInitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto jtmb        = llvm::orc::JITTargetMachineBuilder::detectHost();
  auto data_layout = jtmb->getDefaultDataLayoutForTarget();

  return std::unique_ptr<SimpleOrcJit>(new SimpleOrcJit(std::move(*jtmb), std::move(*data_layout)));
  // return std::make_unique<SimpleOrcJit>(std::move(*jtmb),
  //                                      std::move(*data_layout));
}

void SimpleOrcJit::AddModule(std::unique_ptr<llvm::Module> module, bool optimize) {
  if (optimize) {
    auto fpm = std::make_unique<llvm::legacy::FunctionPassManager>(module.get());

    fpm->add(llvm::createInstructionCombiningPass());
    fpm->add(llvm::createReassociatePass());
    fpm->add(llvm::createGVNPass());
    fpm->add(llvm::createCFGSimplificationPass());
    fpm->doInitialization();

    for (auto &fn : *module) {
      fpm->run(fn);
      fn.print(llvm::outs(), {});
    }
  }

  auto unused = compile_layer_.add(main_jd_, llvm::orc::ThreadSafeModule(std::move(module), context_));
}

void SimpleOrcJit::Link(const lang::Module &module, bool optimize) {
  std::string runtime_ir(backends::kRuntimeLlvmIr);
  llvm::SMDiagnostic error;
  auto m = llvm::parseAssemblyString(runtime_ir, error, context());
  auto b = std::make_unique<llvm::IRBuilder<>>(context());

  auto ir_emitter = std::make_unique<CodeGenLLVM>(m.get(), b.get());
  for (auto &buffer : module->buffers) {
    auto expr = runtime::BufferCreate(buffer);
    ir_emitter->Visit(&expr);
  }

  for (auto &fn : module.functions()) {
    ir::Expr fn_expr(fn);
    ir_emitter->Visit(&fn_expr);
  }

  AddModule(std::move(m), optimize);
}

}  // namespace backends
}  // namespace cinn
