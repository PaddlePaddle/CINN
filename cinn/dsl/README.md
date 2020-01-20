# Design of CINN/DSL
This module is a simple DSL defined in CINN project. 
The DSL module aims to represent the overall computation in a hardware indenpendent way.

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