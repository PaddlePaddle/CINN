func @dense_shape0() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32.2 [12:i64, 23:i64]
  dt.fill_tensor_with_constant.f32 %a 0.1:f32

  cinn.return
}
