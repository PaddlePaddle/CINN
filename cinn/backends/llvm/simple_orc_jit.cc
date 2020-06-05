#include "cinn/backends/llvm/simple_orc_jit.h"

#include <llvm/ADT/Triple.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <cmath>
#include <memory>
#include <utility>

#include "cinn/backends/llvm/cinn_runtime_llvm_ir.h"
#include "cinn/backends/llvm/codegen_llvm.h"
#include "cinn/backends/llvm/llvm_util.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

SimpleOrcJit::SimpleOrcJit(llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout data_layout)
    : object_layer_(execution_session_, std::bind(std::make_unique<llvm::SectionMemoryManager>)),
      data_layout_(std::move(data_layout)),
      mangle_(execution_session_, data_layout_),
      compile_layer_(
          execution_session_, object_layer_, std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb))),
      context_(std::make_unique<llvm::LLVMContext>()) {
  main_jd_ = execution_session_.getJITDylibByName("<main>");
  if (!main_jd_) {
    main_jd_ = &execution_session_.createJITDylib("<main>");
  }
  main_jd_->addGenerator(
      llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(data_layout_.getGlobalPrefix())));
  RegisterRuntimeSymbols();
}

/*static*/ std::unique_ptr<SimpleOrcJit> SimpleOrcJit::Create() {
  LLVMInitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto jtmb        = llvm::orc::JITTargetMachineBuilder::detectHost();
  auto data_layout = jtmb->getDefaultDataLayoutForTarget();

  std::unique_ptr<SimpleOrcJit> compiler(new SimpleOrcJit(std::move(*jtmb), std::move(*data_layout)));

  auto compile_function_creator = [&](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    auto tm = jtmb.createTargetMachine();
    if (!tm) return tm.takeError();
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*tm));
  };

  auto object_layer_creator = [&](llvm::orc::ExecutionSession &session, const llvm::Triple &triple) {
    auto object_layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        session, []() { return std::make_unique<llvm::SectionMemoryManager>(); });
    llvm::orc::JITDylib *main_jd = session.getJITDylibByName("<main>");
    if (!main_jd) {
      main_jd = &session.createJITDylib("<main>");
    }
    main_jd->addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(data_layout->getGlobalPrefix())));
    return object_layer;
  };
  compiler->jit_ = llvm::cantFail(llvm::orc::LLJITBuilder()
                                      .setCompileFunctionCreator(compile_function_creator)
                                      .setObjectLinkingLayerCreator(object_layer_creator)
                                      .create());

  return compiler;
}

bool SimpleOrcJit::SetupTargetTriple(llvm::Module *module) {
  auto target_triple = llvm::sys::getDefaultTargetTriple();
  std::string error_msg;
  auto target = llvm::TargetRegistry::lookupTarget(target_triple, error_msg);
  if (!target) {
    LOG(ERROR) << "no target: " << error_msg;
    return true;
  }

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(target_triple, "generic", "", {}, {}));
  module->setDataLayout(machine->createDataLayout());
  module->setTargetTriple(target_triple);
  return false;
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
    }
  }

  llvm::orc::ThreadSafeModule tsm(std::move(module), context_);
  // llvm::cantFail(jit_->addIRModule(*main_jd_, std::move(tsm)));
  llvm::cantFail(jit_->addIRModule(std::move(tsm)));
}

void SimpleOrcJit::Link(const lang::Module &module, bool optimize, bool dump) {
  std::string runtime_ir(backends::kRuntimeLlvmIr);
  llvm::SMDiagnostic error;
  auto m = llvm::parseAssemblyString(runtime_ir, error, context());
  auto b = std::make_unique<llvm::IRBuilder<>>(context());

  auto ir_emitter = std::make_unique<CodeGenLLVM>(m.get(), b.get());
  for (auto &buffer : module->buffers) {
    auto expr = runtime::BufferCreate(buffer.as_buffer_ref());
    ir_emitter->Visit(&expr);
  }

  for (auto &fn : module.functions()) {
    VLOG(1) << "JIT Linking function [" << fn->name << "]";
    ir::Expr fn_expr(fn);

    auto *fn_ = ir_emitter->Visit(&fn_expr);

    if (dump) LOG(INFO) << DumpToString(*fn_);
  }

  if (dump) {
    LOG(INFO) << "======= dump jit lib[begin] ==========";
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    main_jd_->dump(os);
    os.flush();
    LOG(INFO) << buffer;
  }

  AddModule(std::move(m), optimize);

  if (dump) {
    LOG(INFO) << "======= dump jit lib[end] ==========";
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    main_jd_->dump(os);
    os.flush();
    LOG(INFO) << buffer;
  }

  for (auto &fn : module.functions()) {
    CHECK(Lookup(fn->name)) << "CINN fn " << fn->name << " not found after link";
  }
}

void *SimpleOrcJit::Lookup(std::string_view name) {
  std::lock_guard<std::mutex> lock(mu_);

  if (auto symbol = jit_->lookup(AsStringRef(name))) {
    return reinterpret_cast<void *>(symbol->getAddress());
  }

  return nullptr;
}

void SimpleOrcJit::RegisterRuntimeSymbols() {
  llvm::orc::SymbolMap symbols;
  auto &registry = RuntimeSymbolRegistry::Global();

  for (const auto &[k, v] : registry.All()) {
    VLOG(1) << "Insert runtime symbol " << k << " to JIT system";
    symbols.insert({mangle_(k), {llvm::pointerToJITTargetAddress(v), llvm::JITSymbolFlags::None}});
  }

  auto error = main_jd_->define(llvm::orc::absoluteSymbols(std::move(symbols)));
  CHECK(!error) << "JIT add runtime symbols failed!";
}

namespace {
bool RegisterKnownSymbols() {
  auto &registry = RuntimeSymbolRegistry::Global();

  registry.Register("sinf", reinterpret_cast<void *>(&sinf));
  registry.Register("sin", reinterpret_cast<void *>(static_cast<double (*)(double)>(&sin)));

  registry.Register("cosf", reinterpret_cast<void *>(&cosf));
  registry.Register("cos", reinterpret_cast<void *>(static_cast<double (*)(double)>(&cos)));
  return true;
}

[[maybe_unused]] bool unused = RegisterKnownSymbols();
}  // namespace

}  // namespace backends
}  // namespace cinn
