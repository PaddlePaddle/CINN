// CHECK: @paddle_func
func @paddle_func() -> () {
  %input = dt.create_uninit_tensor.f32 [3, 5] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !cinn.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [5, 4] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%w : !cinn.tensor<X86, NCHW, F32>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [3, 4] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias : !cinn.tensor<X86, NCHW, F32>) {value=3.0:f32}

  %out = dt.create_uninit_tensor.f32 [3, 4] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out : !cinn.tensor<X86, NCHW, F32>) {value=0.0:f32}

  // test external.matmul
  "external.matmul"(%input, %w, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  // CHECK: tensor: shape=shape[3,4], values=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
  dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  // test external.elementwise_add
  "external.elementwise_add"(%out, %bias, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  // CHECK: tensor: shape=shape[3,4], values=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
  dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  // test external.sigmoid
  "external.sigmoid"(%out, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  // CHECK: tensor: shape=shape[3,4], values=[0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998]
  dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  cinn.return
}
