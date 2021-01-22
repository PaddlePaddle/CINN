// CHECK-LABEL: build_tensor1
func @build_tensor1() {
  %a = dt.create_uninit_tensor.f32 [3:i64, 4:i64] -> !cinn.tensor<x86, NCHW, f32>
  //dt.fill_tensor_with_constant.f32 %a : !cinn.tensor<X86, NCHW, f32> 1.0 : f32
  // CHECK: tensor: shape=shape[3,4], values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  //"dt.print_tensor"(%a) : (!cinn.tensor<X86, NCHW, F32>) -> ()

  cinn.return
}
