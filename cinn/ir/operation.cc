#include "cinn/ir/operation.h"

namespace cinn {
namespace ir {

Operation ExternOp::Make(std::string name,
                         std::string tag,
                         std::map<std::string, IrNodeRef> attrs,
                         std::vector<Tensor> inputs,
                         std::vector<Buffer> input_placeholders,
                         std::vector<Buffer> output_placeholders,
                         Stmt body) {
  auto n   = common::make_shared<ExternOp>();
  n->name  = std::move(name);
  n->tag   = std::move(tag);
  n->attrs = std::move(attrs);
  CHECK_EQ(inputs.size(), input_placeholders.size());

  n->inputs              = std::move(inputs);
  n->input_placeholders  = std::move(input_placeholders);
  n->output_placeholders = std::move(output_placeholders);
  n->body                = std::move(body);
  return Operation(n);
}

}  // namespace ir
}  // namespace cinn