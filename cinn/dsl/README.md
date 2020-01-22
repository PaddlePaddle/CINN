# Design of CINN/DSL
This module is a simple DSL defined in CINN project. 
The DSL module aims to represent the overall computation in a hardware indenpendent way.

## Concepts
### Object
All the mutable elements in CINN are `Object`.
### Shared
The `Shared` objects are reference-count-self-contained container, which is similar to the `std::shared_ptr`.

One can pass a `Shared` object by passing a pointer and the consumer object should store it in a local `Shared` member variable.

## Some examples
A matrix multiplication

```c++
// Declare some iterator variables.
Var i, j, k;
Placeholder<float> A({M, K}), B({K, N});

Tensor C = Compute({M, N}/*output shape*/, 
        [](Var i, Var j, Var k) {
            return ReduceSum(A(i,k) * B(k, j), k);
        }, "C");
        
Schedule s = CreateSchedule(C.op);
auto func = Build(s, [A, B, C], target=target, name="matmul");

func(a, b, c);
```

