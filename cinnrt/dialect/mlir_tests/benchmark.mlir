// CHECK-LABEL: @benchmark
func @benchmark() {
  cinn.benchmark "add.f32"() duration_secs = 1, max_count = 10, num_warmup_runs = 0
  {
    %0 = cinn.constant.f32 1.0
    %1 = cinn.constant.f32 2.0
    %res = "cinn.add.f32"(%0, %1) : (f32, f32) -> f32
    "cinn.print.f32"(%res) : (f32) -> ()
    cinn.return %res : f32
  }
  cinn.return
}
