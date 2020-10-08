#include "cinn/dialect/opt.h"

#include <mlir/Support/MlirOptMain.h>

#include "cinn/dialect/init_cinn_dialects.h"

int main(int argc, char** argv) { cinn::dialect::RegisterCinnDialects(); }
