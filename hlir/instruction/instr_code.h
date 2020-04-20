#pragma once

const int kHlirVarientArgusCode = -1;
// clang-format off

#define INSTR_CODE_FOR_ALL(op__)             \
    op__(Abs, "abs", 1)                      \
    op__(Add, "add", 2)                      \
    op__(Sub, "sub", 2)                      \
    op__(Mul, "mul", 2)                      \
    op__(Div, "div", 2)                      \
    op__(Not, "not", 1)                      \
    op__(And, "and", 2)                      \
    op__(Or, "or", 2)                        \
    op__(Tanh, "tanh", 1)                    \
    op__(Sigmoid, "sigmoid", 1)              \
    op__(Floor, "floor", 1)                  \
    op__(Ceil, "ceil", 1)                    \
    op__(Compare, "compare", 2)              \
    op__(Parameter, "parameter", 0)          \
    op__(Reduce, "reduce", 0)                \
    op__(Dot, "dot", 2)                      \
    op__(Broadcast, "broadcast", 1)          \
    op__(Transpose, "transpose", 1)          \
    op__(Call, "all", kHlirVarientArgusCode)

// clang-format on

//! Code of all the operation supports in instructions.
#define __(a__, b__, c__) a__,
enum class InstrCode { Unknown = -1, INSTR_CODE_FOR_ALL(__) };
#undef __
