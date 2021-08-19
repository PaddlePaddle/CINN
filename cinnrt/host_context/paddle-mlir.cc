#include <fstream>
#include <iostream>
#include <string>

#include "cinnrt/common/global.h"
#include "cinnrt/common/string.h"
#include "cinnrt/dialect/basic_kernels.h"
#include "cinnrt/dialect/cinn_base.h"
#include "cinnrt/dialect/dense_tensor.h"
#include "cinnrt/dialect/init_cinn_dialects.h"
#include "cinnrt/dialect/pd_ops.h"
#include "cinnrt/dialect/tensor_shape.h"
#include "cinnrt/paddle/model_parser.h"
//#include "cinnrt/framework/framework.pb.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

using namespace std;
using Scope = cinnrt::paddle::Scope;
// using ProgramDesc = cinn::frontend::paddle::cpp::ProgramDesc;
using Target = cinnrt::common::Target;

namespace cl = llvm::cl;
static cl::opt<std::string> paddleModelDir(cl::Positional,
                                           cl::desc("<paddle model dir>"),
                                           cl::init(""),
                                           cl::value_desc("model dir"));

//////////////////debug code begin//////////////
llvm::raw_ostream &printIndent(int indent = 0) {
  for (int i = 0; i < indent; ++i) llvm::outs() << "    ";
  return llvm::outs();
}

void printOperation(mlir::Operation *op, int indent);
void printRegion(mlir::Region &region, int indent);
void printBlock(mlir::Block &block, int indent);

void printOperation(mlir::Operation *op, int indent) {
  llvm::Optional<mlir::ModuleOp> module_op = llvm::None;
  if (llvm::isa<mlir::ModuleOp>(op)) module_op = llvm::dyn_cast<mlir::ModuleOp>(op);
  llvm::Optional<mlir::FuncOp> func_op = llvm::None;
  if (llvm::isa<mlir::FuncOp>(op)) func_op = llvm::dyn_cast<mlir::FuncOp>(op);

  printIndent(indent) << "op: '" << op->getName();
  // This getName is inherited from Operation::getName
  if (module_op) {
    printIndent() << "@" << module_op->getName();
  }
  // This getName is inherited from SymbolOpInterfaceTrait::getName,
  // which return value of "sym_name" in ModuleOp or FuncOp attributes.
  if (func_op) {
    printIndent() << "@" << func_op->getName();
  }
  printIndent() << "' with " << op->getNumOperands() << " operands"
                << ", " << op->getNumResults() << " results"
                << ", " << op->getAttrs().size() << " attributes"
                << ", " << op->getNumRegions() << " regions"
                << ", " << op->getNumSuccessors() << " successors\n";
  if (!op->getAttrs().empty()) {
    printIndent(indent) << op->getAttrs().size() << " attributes:\n";
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      printIndent(indent + 1) << "- {" << attr.first << " : " << attr.second << "}\n";
    }
  }

  if (op->getNumRegions() > 0) {
    printIndent(indent) << op->getNumRegions() << " nested regions:\n";
    for (mlir::Region &region : op->getRegions()) {
      printRegion(region, indent + 1);
    }
  }
}

void printRegion(mlir::Region &region, int indent) {
  printIndent(indent) << "Region with " << region.getBlocks().size() << " blocks:\n";
  for (mlir::Block &block : region.getBlocks()) {
    printBlock(block, indent + 1);
  }
}

void dumpBlock(mlir::Block &block) {
  cout << "dump block begin" << endl;
  llvm::raw_ostream &os     = llvm::errs();
  mlir::Operation *parentOp = block.getParentOp();
  if (!parentOp) {
    os << "<<UNLINKED BLOCK>>\n";
    return;
  }
  while (auto *nextOp = parentOp->getParentOp()) parentOp = nextOp;
  mlir::AsmState state(parentOp);
  // block.print(os, state);
  // mlir::OperationPrinter(os, /*flags=*/llvm::None, state.getImpl()).print(&block, false);
  auto range = llvm::make_range(block.getOperations().begin(), std::prev(block.getOperations().end(), 0));
  for (auto &op : range) {
    // print(&op);
    // os << newLine;
  }
  cout << endl << "dump block end" << endl;
}

void printBlock(mlir::Block &block, int indent) {
  // dumpBlock(block);
  printIndent(indent) << "Block with " << block.getNumArguments() << " arguments"
                      << ", " << block.getNumSuccessors() << " successors"
                      << ", " << block.getOperations().size() << " operations\n";

  for (mlir::Operation &operation : block.getOperations()) {
    // cout << "dump operation begin" << endl;
    ////operation.dump();
    // operation.print(llvm::errs(), mlir::OpPrintingFlags().useLocalScope());
    // cout << "dump operation end" << endl;
    printOperation(&operation, indent + 1);
  }
}
//////////////////debug code end//////////////

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "paddle-mlir");
  if (paddleModelDir.empty()) {
    cout << "ERROR: paddle model path can't be empty." << endl;
    return 1;
  }

  // Create ProgramDesc
  // ProgramDesc program;
  Scope scope;
  Target target;
  target.arch            = Target::Arch ::X86;
  target.bits            = Target::Bit ::k32;
  target.os              = Target::OS ::Linux;
  std::string model_path = paddleModelDir + "/__model__";

  auto program_proto = *cinnrt::paddle::LoadProgram(model_path);
  // cinn::frontend::paddle::pb::ProgramDesc pb_prog_desc(&program_proto);
  // cinn::frontend::paddle::TransformProgramDescAnyToCpp(pb_prog_desc, &program);

  for (auto &block_desc : program_proto.blocks()) {
    for (auto &op_desc : block_desc.ops()) {
      cout << "find op: " << op_desc.type() << " " << op_desc.attrs(0).name() << endl;
      cout << op_desc.type() << "(";
      for (int var_idx = 0; var_idx < op_desc.inputs_size(); ++var_idx) {
        auto &var = op_desc.inputs(var_idx);
        cout << var.parameter() << ":";
        for (auto &arg : var.arguments()) {
          cout << arg;
        }
        if (var_idx < op_desc.inputs_size() - 1) cout << ", ";
      }
      cout << ") -> (";

      for (int var_idx = 0; var_idx < op_desc.outputs_size(); ++var_idx) {
        auto &var = op_desc.outputs(var_idx);
        cout << var.parameter() << ":";
        for (auto &arg : var.arguments()) {
          cout << arg;
        }
        if (var_idx < op_desc.outputs_size() - 1) cout << ", ";
      }
      cout << ")" << endl;
      ;
    }
  }

  auto main_block = program_proto.blocks(0);
  for (auto &var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) continue;
    std::string param_path = paddleModelDir + "/" + var.name();
    std::ifstream param_file(param_path, std::ios::binary);
    switch (var.type().type()) {
      case paddle::framework::proto::VarType_Type_LOD_TENSOR: {
        auto var_name = cinnrt::common::TransValidVarName(var.name());
        std::cout << "var name: " << var.name() << " " << var_name << std::endl;
        auto *_var = scope.Var<cinnrt::paddle::Tensor>(var_name);
        cinnrt::paddle::LoadLoDTensor(param_file, _var, target);
        auto tensor     = scope.GetTensor(var_name);
        auto *src_data  = tensor->data<float>();
        auto &cinn_type = tensor->type();
        std::vector<int64_t> shape;
        for (int dim : tensor->shape().data()) shape.push_back(dim);
        break;
      }
      default:
        std::cout << "unknown weight type" << std::endl;
        break;
    }
  }

  mlir::MLIRContext *context = cinnrt::Global::getMLIRContext();
  context->allowUnregisteredDialects();
  auto &registry = context->getDialectRegistry();
  // cinnrt::RegisterCinnDialects(registry);
  // registry.insert<mlir::StandardOpsDialect>();
  context->getOrLoadDialect<mlir::StandardOpsDialect>();
  context->getOrLoadDialect<cinnrt::dialect::CINNDialect>();
  context->getOrLoadDialect<cinnrt::ts::TensorShapeDialect>();
  context->getOrLoadDialect<cinnrt::dt::DTDialect>();
  context->getOrLoadDialect<mlir::PD::PaddleDialect>();

  for (auto dialect : context->getLoadedDialects()) {
    cout << "loaded dialect: " << dialect->getNamespace().str() << endl;
  }

  mlir::OpBuilder builder(context);
  mlir::OwningModuleRef module_ref;
  module_ref = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  using cinnrt::dt::LayoutType;
  using cinnrt::dt::PrecisionType;
  using cinnrt::dt::TargetType;
  using cinnrt::dt::TensorType;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  // operandTypes.push_back(builder.getF32Type());
  // input tensor
  operandTypes.push_back(TensorType::get(TargetType::X86, LayoutType::NCHW, PrecisionType::F32));
  operandTypes.push_back(cinnrt::dt::TensorMapType::get(context));

  llvm::SmallVector<mlir::Type, 4> resultTypes;
  // return tensor
  resultTypes.push_back(TensorType::get(TargetType::X86, LayoutType::NCHW, PrecisionType::F32));
  mlir::FuncOp mainFunc = mlir::FuncOp::create(mlir::UnknownLoc::get(context),
                                               /*name=*/"predict",
                                               /*type=*/builder.getFunctionType(operandTypes, resultTypes),
                                               /*attrs=*/{});
  module_ref->push_back(mainFunc);
  // printOperation(module_ref->getOperation(), 0);
  cout << "mainFunc.isExternal: " << mainFunc.isExternal() << endl;
  cout << "mainFunc.getNumRegions: " << mainFunc.getOperation()->getNumRegions() << endl;
  // mlir::Region* region = new mlir::Region();
  // mainFunc.getCallableRegion()->takeBody(*region);
  cout << "getRegion: " << &mainFunc.getOperation()->getRegion(0) << endl;
  cout << "getCallableRegion: " << mainFunc.getCallableRegion() << endl;
  cout << "getBody: " << &mainFunc.getBody() << endl;
  cout << "getBody empty: " << mainFunc.getBody().empty() << endl;
  mlir::Block *block = new mlir::Block();
  // mainFunc.getBody().push_back(block);
  mainFunc.addEntryBlock();

  builder.setInsertionPointToStart(&mainFunc.getBody().back());
  // auto op = builder.create<cinnrt::dialect::ReturnOp>(mlir::UnknownLoc());
  llvm::SmallVector<mlir::Value, 4> retVals;
  mlir::Location loc = mlir::UnknownLoc::get(context);
  mlir::OperationState state(mlir::UnknownLoc::get(context), mlir::ReturnOp::getOperationName());
  // mlir::ReturnOp::build(builder, state, retVals);
  // auto op = builder.create<cinnrt::dialect::ReturnOp>(mlir::UnknownLoc::get(context));

  ::mlir::Type outputType = TensorType::get(TargetType::X86, LayoutType::NCHW, PrecisionType::F32);
  ::mlir::Value map       = mainFunc.getArgument(0);
  ::mlir::StringAttr name;
  // %w = dt.get_param
  name     = builder.getStringAttr("create_parameter_0.w_0");
  auto op1 = builder.create<cinnrt::dt::GetParamOp>(mlir::UnknownLoc::get(context), outputType, map, name);
  // %bias = dt.get_param
  name     = builder.getStringAttr("create_parameter_1.w_0");
  auto op2 = builder.create<cinnrt::dt::GetParamOp>(mlir::UnknownLoc::get(context), outputType, map, name);
  // %out = dt.create_uninit_tensor.f32
  ::mlir::ArrayAttr array =
      builder.getI32ArrayAttr(llvm::makeArrayRef({1, 1}));  //::mlir::ArrayAttr array = builder.getI32ArrayAttr({1, 1});
  auto op3 = builder.create<cinnrt::dt::CreateUninitTensorOp_f32>(mlir::UnknownLoc::get(context), outputType, array);
  // "external.matmul"
  {
    ::mlir::OperationName op_name("external.matmul", context);
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.push_back(mainFunc.getArgument(0));
    operands.push_back(op1.getOperation()->getResult(0));
    operands.push_back(op3.getOperation()->getResult(0));
    llvm::SmallVector<mlir::Type, 4> resultTypes1;
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    ::mlir::Operation *customop = ::mlir::Operation::create(loc, op_name, resultTypes1, operands, {});
    builder.insert(customop);
  }
  // "external.elementwise_add"
  {
    ::mlir::OperationName op_name("external.elementwise_add", context);
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.push_back(op3.getOperation()->getResult(0));
    operands.push_back(op2.getOperation()->getResult(0));
    operands.push_back(op3.getOperation()->getResult(0));
    llvm::SmallVector<mlir::Type, 4> resultTypes1;
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    ::mlir::Operation *customop = ::mlir::Operation::create(loc, op_name, resultTypes1, operands, {});
    builder.insert(customop);
  }
  // "external.sigmoid"
  {
    ::mlir::OperationName op_name("external.sigmoid", context);
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.push_back(op3.getOperation()->getResult(0));
    operands.push_back(op3.getOperation()->getResult(0));
    llvm::SmallVector<mlir::Type, 4> resultTypes1;
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    ::mlir::Operation *customop = ::mlir::Operation::create(loc, op_name, resultTypes1, operands, {});
    builder.insert(customop);
  }
  // return
  {
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.push_back(op3.getOperation()->getResult(0));
    auto op_return = builder.create<cinnrt::dialect::ReturnOp>(mlir::UnknownLoc::get(context), operands);
  }
  // auto op = builder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(context), retVals);
  // cout << "block->push_back begin" << endl;
  // block->push_back(op);
  // cout << "block->push_back end" << endl;

  printOperation(module_ref->getOperation(), 0);
  cout << "module dump begin" << endl;
  module_ref->dump();
  cout << endl << "module dump end" << endl;
  cout << endl;
  // cout << "model contains " << program_proto.blocks_size() << " blocks." << endl;

  // for (auto &block_desc : program_proto.blocks()) {
  //    for (auto &op_desc : block_desc.ops()) {
  //        cout << "find op: " << op_desc.type() << " " << op_desc.attrs(0).name() << endl;
  //        for (auto &var : op_desc.inputs()) {
  //            cout << var.parameter() << " " << var.arguments() << endl;
  //        }
  //    }
  //}

  // if (!config_.model_from_memory()) {
  //  std::string pb_content;
  //  // Read binary
  //  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  //  PADDLE_ENFORCE_EQ(
  //      static_cast<bool>(fin.is_open()), true,
  //      platform::errors::NotFound(
  //          "Cannot open file %s, please confirm whether the file is normal.",
  //          filename));
  //  fin.seekg(0, std::ios::end);
  //  pb_content.resize(fin.tellg());
  //  fin.seekg(0, std::ios::beg);
  //  fin.read(&(pb_content.at(0)), pb_content.size());
  //  fin.close();

  //  proto.ParseFromString(pb_content);
  //} else {
  //  proto.ParseFromString(config_.prog_file());
  //}
}
