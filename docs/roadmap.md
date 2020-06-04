# v0.1-alpha
This release will focus on the core framework, it plans to support the following features:

- [x] CPU X86: high performance CodeGenC_X86 and CodeGenLLVM_X86 backend,
- [ ] CUDA: the GPU related schedule, the base performance,
- The core framework:
  - [ ] a well-encapsulated JIT framework that can call CINN expression or external functions
  - [ ] full support for extern-Call, the ability to trigger the MKL or CUBLAS functions,
  - [ ] a computation definition and schedule framework to provide the ability to define new algorithm and performance automatically tune.
- API
  - [ ] python api for HLIR and CINN(optional)
  - [ ] C++ APIs
- Documents to introduce
  - [ ] the basic concepts
  - [ ] the architecture
  - [ ] the usage of the APIs
  - [ ] example to develop new HLIR/primitive instructions
- Some basic primitives:
  - [ ] the actives,
  - [ ] Dot,
  - [ ] some binary ones, such as Add, Sub and so on,
  - [ ] Conv
- [ ] provide a real model and its benchmark
