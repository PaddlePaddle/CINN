// CHECK-LABEL: test_tensor_type
func @test_tensor_type() -> !cinn.tensor_map {
  %path = cinn.get_string("/cinn/benchmark/paddle-inference/Paddle-Inference-Demo/c++/fc/fc_1.8")
  %map = dt.load_tensors(%path)
  cinn.return %map : !cinn.tensor_map
}

func @predict() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !cinn.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.get_tensor("create_parameter_0.w_0") -> !cinn.tensor<X86, NCHW, F32>
  %bias = dt.get_tensor("create_parameter_1.w_0") -> !cinn.tensor<X86, NCHW, F32>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !cinn.tensor<X86, NCHW, F32>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()

  dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  cinn.return
}
