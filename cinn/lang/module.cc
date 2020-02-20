#include "cinn/lang/module.h"

namespace cinn {
namespace lang {

/**
 * Content of a module.
 */
struct _Module_ : Object {
  std::string name;
  Target target;
  std::vector<ir::Buffer> buffers;
  std::vector<ir::PackedFunc> functions;
  std::vector<Module> submodules;
};

}  // namespace lang
}  // namespace cinn
