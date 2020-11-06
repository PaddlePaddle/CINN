func @basic_f32() -> f32 {
  %v0 = cinn.constant.f32 1.0
  %v1 = cinn.constant.f32 2.0
  %value = "cinn.add.f32"(%v0, %v1) : (f32, f32) -> f32

  "cinn.print.f32"(%v0) : (f32) -> ()

  cinn.return %value : f32
}
