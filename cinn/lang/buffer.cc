#include "cinn/lang/buffer.h"

#include "cinn/ir/buffer.h"

namespace cinn {
namespace lang {

using ir::_Buffer_;

Buffer::Buffer(Type type, const std::string& name) {
  buffer_        = _Buffer_::Make();
  buffer_->dtype = type;
  buffer_->set_type(type_of<cinn_buffer_t*>());
  buffer_->elem_offset = Expr(0);
  if (!name.empty()) {
    buffer_->name = name;
  }
  buffer_->target = common::DefaultHostTarget();
}

}  // namespace lang
}  // namespace cinn
