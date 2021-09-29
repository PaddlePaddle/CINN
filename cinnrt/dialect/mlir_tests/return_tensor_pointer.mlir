// CHECK-LABEL: @return_tensor_pointer
func @return_tensor_pointer() {
  %input = dt.create_uninit_tensor.f32 [2, 2] -> !cinn.tensor<X86, NCHW, F32>
  %out = dt.fill_tensor_and_return.f32 (%input: !cinn.tensor<X86, NCHW, F32>) {value=1.0:f32} -> !cinn.tensor<X86, NCHW, F32>
  // CHECK-LABEL: tensor: shape=shape[2,2], values=[1, 1, 1, 1]
  dt.print_tensor (%input : !cinn.tensor<X86, NCHW, F32>)
  // CHECK-LABEL: tensor: shape=shape[2,2], values=[1, 1, 1, 1]
  "external.print_tensor_pointer"(%out) {}: (!cinn.tensor<X86, NCHW, F32>) -> ()

  cinn.return
}
