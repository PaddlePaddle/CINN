# Design of CINN

## Multi-layer architecture
To enable the compiler to handle the general DNN tasks, a multi-layer system is
introduced.

The layers are as follows

1. NN compatible layer
    - Add operator wrapper for DNN platform such as PaddlePaddle or TensorFlow.
2. Virtual graph layer
    - Add virtual node and graph to utilize both the compiler and third-party computational library such as CUDNN or MKLDNN.
3. DSL layer
    - Export a friendly domain language to programming with the underlying compiler.
4. Compiler layer
    - A NN compiler which can optimize affine forloop.
    
## DSL layer

**Matrix multiplication with blocking**

```c++
Var i("i"), j("j"), k("k");
Constant N("N"), M("M"), K("K");

PlaceHolder<float> x("x");
PlaceHolder<float> y("x");

Tensor C = compute({M, N}/*dims*/, [&](Var i, Var j, Var k){
    return x(i,k) * y(k,j);
});

Schedule S = ComputeSchedule({C}/*outputs*/, {A,B}/*inputs*/);
{ // schedule C's computation

    
    // tile i, j with factor 4
    Var i0,i1,j0,j1;
    std::tie(i0,i1,j0,j1) = S[C].tile(S[C].axis("i"), S[C].axis("j"), 4, 4);
    
    // tile k with factor 4
    Var k0,k1;
    std::tie(k0,k1) = S[C].split(k, 4);
    
    S[C].reorder(i0,j0,k0,k1,i1,j1); // swap(i1,j0)
}
```

**Matrix with Vectorization**

```c++
Schedule S = ComputeSchedule({C}, {A,B});

Var k0,k1;
std::tie(k0,k1) = S[C].split(k,4);
Var x0,x1,y0,y1;
std::tie(x0, x1, y0, y1) = S[C].tile(x, y, 4, 4);

S[C].reorder(x0, y0, k0, k1, x1, y1);

S[C].vectorize(y1);
```

**Matrix with Packing**

```c++
Tensor packedB = compute((N/bn, K, bn), [&](Var i, Var j, Var k) {
    return B(j, i*bn+k);
});

Tensor C = compute({M, N}, [&](Var i, Var j, Var k) {
    // reduce sum(need initialize)
    return sum(A(i,k) * packedB(y/bn, k, y%bn), k);
});

Schedule S = compute_schedule({C}, {A,B});

Var i0,j0,i1,j1;
Var k0,k1;
std::tie(i0,i1,j0,j1) = S[C].tile(S[C].axis(0), S.axis(1), 4, 4);
std::tie(k0,k1) = S[C].split(S[C].axis(k, 4);

S[C].reorder(i0, j0, k0, i1, k1, j1);
S[C].vectorize(j1);

{
    Var i, j, k;
    std::tie(i,j,k) = S[packedB].axis();
    S[packedB].vectorize(k);
    S[packedB].parallel(i);
}
```

## Compiler Layer

### IR
The IR is similar to Halide.

#### Basic elements
The IR has following basic elements:

- Expr, the expression in the IR(which represents a value or returns a value).
- Stmt, the statement in the IR.
- Tensor (the input or temporary value)
- Buffer (the memory buffer)

### Tensor
Tensor represents the input or temporary variable.

Each tensor is assigned a buffer by default, but `store_in` can change the relation.


### Polyhedral usage
The polyhedral technology is used to simplify the forloop analysis and transform.

### schedule
The original tensor-based computation forms a SSA graph.

Each tensor is assign a `Stage`, which is the basic schedule element.

A stage has a domain(isl.Set) and a schedule(isl.Map), all the schedule is performed on them.

### Stage
The stage supports multiple operations.

in CodeGen, each stage(noninline or fuse_with) will generte a isl.ast.
