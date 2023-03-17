## Description

Please describe this pull request.

### 算子类型

- [ ] ElementWise：The relation between input tensor index and output tensor index is one-to-one correspondence.
- [ ] Broadcast：The relation between input tensor index and output tensor index is one-to-many correspondence.
- [ ] Injective：Injective operator, we can always injectively map a output axis to a input axis.
- [ ] Reduction：The relation between input tensor index and output tensor index is many-to-one correspondence.
- [ ] OutFusible：Complex operation, can still fuse one-to-one operations into its output.
- [ ] kNonFusible：Operation that cannot fuse anything.

### OpMapper

- [ ] Is this operator an OpMapper? If yes, please paste the OpMaker code path in Paddle. (Github link is preferred)

## Test Cases Checklist

### Tensor Shape

- [ ] 1D Tensor
- [ ] 2D Tensor
- [ ] 3D Tensor
- [ ] 4D Tensor

#### special shape

- [ ] One dimension of the tensor is 1.
- [ ] One dimension of the tensor is less than 1024.
- [ ] One dimension of the tensor is greater than 1024.
- [ ] All dimensions of the tensor are 1

### Tensor Dtype

- [ ] int32
- [ ] int64
- [ ] float16
- [ ] float32
- [ ] float64

### Broadcast

- [ ] Does the input tensor support broadcasting? 
- [ ] Broadcasting test cases

### Operator Attributes

Test cases for operator attributes.

- [ ] Attribute: attribute type - possible values
    - [ ] Possible value 1
    - [ ] Possible value 2
    - [ ] Possible value 3
- [ ] Use OpTestHelper to test the Cartesian product combination of the above attributes.

