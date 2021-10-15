// CHECK-LABEL: @main
func @main() {
  %a = "pd.Feed"() : () -> tensor<?xf32>
  %b = "pd.Feed"() : () -> tensor<?xf32>
  %bias = "pd.Feed"() : () -> tensor<?xf32>

  %c = "pd.Matmul"(%a, %b) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d = "pd.ElementwiseAdd"(%c, %bias) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  cinn.return
}
