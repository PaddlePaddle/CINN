def TensorShapeDialect : Dialect {
  let name = "ts";

  let description = [{
    The Tensor Shape dialect.
  }];

  let cppNamespace = "::cinn::dialect";
}
