#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>

namespace cinnrt::hlir::dialect {

class CinnDialect : public ::mlir::Dialect {
 public:
  explicit CinnDialect(::mlir::MLIRContext* ctx);

  //! We should register this function in dialect
  static llvm::StringRef getDialectNamespace() { return "cinn::hlir::dialect"; }
};

}  // namespace cinnrt::hlir::dialect