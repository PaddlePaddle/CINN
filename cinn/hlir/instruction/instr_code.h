#pragma once
#include <ostream>

const int kHlirVarientArgusCode = -1;
// clang-format off

#define INSTR_CODE_FOR_ALL(op__)                           \
    op__(Abs, "abs", 1)                                    \
    op__(Add, "add", 2)                                    \
    op__(Sub, "sub", 2)                                    \
    op__(Mul, "mul", 2)                                    \
    op__(Div, "div", 2)                                    \
    op__(Not, "not", 1)                                    \
    op__(And, "and", 2)                                    \
    op__(Or, "or", 2)                                      \
    op__(Tanh, "tanh", 1)                                  \
    op__(Sigmoid, "sigmoid", 1)                            \
    op__(Floor, "floor", 1)                                \
    op__(Ceil, "ceil", 1)                                  \
    op__(Compare, "compare", 2)                            \
    op__(Parameter, "parameter", 0)                        \
    op__(Input, "input", 0)                                \
    op__(Reduce, "reduce", 0)                              \
    op__(Dot, "dot", 2)                                    \
    op__(Broadcast, "broadcast", 1)                        \
    op__(Transpose, "transpose", 1)                        \
    op__(Constant, "constant", 0)                          \
    op__(Tuple, "tuple", kHlirVarientArgusCode)            \
    op__(TupleGet, "tuple_get", 1)                         \
    op__(Call, "call", kHlirVarientArgusCode)              \
    op__(Conv, "conv", 2)                                  \
    op__(CustomCall, "custom_call", kHlirVarientArgusCode)

// clang-format on

namespace cinn {
namespace hlir {
namespace instruction {

//! Code of all the operation supports in instructions.
#define __(a__, b__, c__) a__,
enum class InstrCode { Unknown = -1, INSTR_CODE_FOR_ALL(__) };
#undef __

static const char* InstrCodeToString(InstrCode code) {
  switch (code) {
#define __(a__, b__, c__) \
  case InstrCode::a__:    \
    return b__;           \
    break;

    INSTR_CODE_FOR_ALL(__);
#undef __

    case InstrCode::Unknown:
      return "unk";
      break;
  }
  return "";
}

static std::ostream& operator<<(std::ostream& os, InstrCode code) {
  os << InstrCodeToString(code);
  return os;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
