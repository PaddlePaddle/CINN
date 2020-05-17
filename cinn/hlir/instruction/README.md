# Instruction layer

The Instruction layer contains the primitive operations of DNNs. 

The SSA makes the analysis and optimization on the higher level possible.

## Demo
```c++
Builder builder;

const int Batch = 10;
const int M = 100;
const int N = 100;

// decalare two input slots: x, y
auto* x = builder.AddInstruction(Instruction::CreateInput({Batch, M, N}), "x");
auto* y = builder.AddInstruction(Instruction::CreateInput({Batch, N, N}), "y");
auto* w = builder.AddInstruction(Instruction::CreateParameter({N, N}, "w"), "w_create");
auto* dot_out = builder.AddInstruction(Instruction::CreateDot({}, x, y), "dot_compute");
auto* add = builder.AddInstruction(Instruction::CreateBinary(InstrCode::Add, dout_out, y), "bias");
```

## Optimize

## Automatic fusion

## Third-party computation library integration

## JIT
