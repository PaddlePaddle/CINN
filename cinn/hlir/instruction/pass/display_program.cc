#include "cinn/hlir/instruction/pass/display_program.h"

#include <glog/raw_logging.h>

#include "cinn/hlir/instruction/module.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace pass {
using ::cinn::utils::StringFormat;

const char* header_splitter = "=========================================================";
const char* footer_splitter = "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";

bool DisplayProgram::Run(Module* module) {
  RAW_LOG(INFO,
          StringFormat("\nHLIR Program:\n%s\n%s%s", header_splitter, module->to_debug_string().c_str(), footer_splitter)
              .c_str());

  return false;
}

std::string_view DisplayProgram::name() const { return name_; }

}  // namespace pass
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
