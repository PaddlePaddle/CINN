#include <iostream>

#include "infrt/host_context/kernel_registry.h"
#include "infrt/host_context/kernel_utils.h"

template <typename T>
T add(T a, T b) {
  return a + b;
}

template <typename T>
T sub(T a, T b) {
  return a - b;
}

template <typename T>
T mul(T a, T b) {
  return a * b;
}

template <typename T>
T div(T a, T b) {
  return a / b;
}

template <typename T>
void print(T a) {
  std::cout << a << std::endl;
}

void RegisterKernels(infrt::host_context::KernelRegistry *registry) {
  // int32
  registry->AddKernel("external.add.i32", CINN_KERNEL(add<int32_t>));
  registry->AddKernel("external.sub.i32", CINN_KERNEL(sub<int32_t>));
  registry->AddKernel("external.mul.i32", CINN_KERNEL(mul<int32_t>));
  registry->AddKernel("external.div.i32", CINN_KERNEL(div<int32_t>));
  registry->AddKernel("external.print.i32", CINN_KERNEL(print<int32_t>));

  // float
  registry->AddKernel("external.add.f32", CINN_KERNEL(add<float>));
  registry->AddKernel("external.sub.f32", CINN_KERNEL(sub<float>));
  registry->AddKernel("external.mul.f32", CINN_KERNEL(mul<float>));
  registry->AddKernel("external.div.f32", CINN_KERNEL(div<float>));
  registry->AddKernel("external.print.f32", CINN_KERNEL(print<float>));
}
