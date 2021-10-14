func @test_fc_fuse_pass()  -> tensor<?xf32> {
  %a = "pd.Feed"() : () -> tensor<?xf32>
  %b = "pd.Feed"() : () -> tensor<?xf32>
  %c = "pd.Feed"() : () -> tensor<?xf32>

  %d = "pd.Matmul"(%a, %b) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e = "pd.ElementwiseAdd"(%d, %c){axis= 1:i32}: (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  cinn.return %e : tensor<?xf32>
}
