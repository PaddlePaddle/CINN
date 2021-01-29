func @basic_f32() -> f32 {
  %v0 = cinn.constant.f32 1.0
  %v1 = cinn.constant.f32 2.0
  %value = "cinn.add.f32"(%v0, %v1) : (f32, f32) -> f32

  "cinn.print.f32"(%v0) : (f32) -> ()

  cinn.return %value : f32
}

// CHECK-LABEL: func @callee.add() -> f32
func @callee.add.f32(%x : f32, %y : f32, %y1 : f32) -> f32 {
  %z = "cinn.add.f32"(%x, %y) : (f32, f32) -> f32
  %z1 = "cinn.add.f32"(%x, %z) : (f32, f32) -> f32
  cinn.return %z1 : f32
}

// CHECK-LABEL: func @caller.add.f32() -> f32
func @caller.add.f32() -> f32 {
  %x = cinn.constant.f32 1.0
  %y = cinn.constant.f32 2.0
  %y1 = cinn.constant.f32 3.0
  %z = cinn.call @callee.add.f32(%x, %y, %y1) : (f32, f32, f32) -> f32

  "cinn.print.f32"(%z) : (f32) -> ()
  cinn.return %z : f32
}
