func @main() -> f32 {
  %v0 = cinn.constant.f32 1.0
  %v1 = cinn.constant.f32 2.0
  %v2 = "cinn.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK: 1
  "cinn.print.f32"(%v0) : (f32) -> ()
  // CHECK: 2
  "cinn.print.f32"(%v1) : (f32) -> ()

  // CHECK: 3
  "cinn.print.f32"(%v2) : (f32) -> ()

  %v3 = "cinn.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  // CHECK: 6
  "cinn.print.f32"(%v3) : (f32) -> ()

  cinn.return %v3 : f32
}
