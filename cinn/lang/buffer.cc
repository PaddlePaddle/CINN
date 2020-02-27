#include "cinn/lang/buffer.h"

#include "cinn/ir/buffer.h"

namespace cinn {
namespace lang {

using ir::_Buffer_;

Buffer::Buffer(const std::string &name) { buffer_ = _Buffer_::Make(name); }

Buffer::Buffer() { buffer_ = _Buffer_::Make(); }

}  // namespace lang
}  // namespace cinn
