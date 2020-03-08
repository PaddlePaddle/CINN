#include "cinn/lang/buffer.h"

#include "cinn/ir/buffer.h"

namespace cinn {
namespace lang {

using ir::_Buffer_;

Buffer::Buffer(Type type, const std::string& name) {
  buffer_ = _Buffer_::Make();
  buffer_->set_type(type);
  buffer_->elem_offset = Expr(0);
  if (!name.empty()) {
    buffer_->name = name;
  }
}

}  // namespace lang
}  // namespace cinn
