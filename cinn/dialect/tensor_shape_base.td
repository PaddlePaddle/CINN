def TensorShapeDialect : Dialect {
  let name = "ts";

  let description = [{
    The Tensor Shape dialect.
  }];

  let cppNamespace = "::cinn::dialect";
}

// Type definitions
def TS_Shape : DialectType<TensorShapeDialect,
                        CPred<"$_self.isa<::cinn::dialect::ShapeType>()">, "!cinn.shape type">,
  BuildableType<"$_builder.getType<::cinn::dialect::ShapeType>()"> {
  let typeDescription = [{
      `!cinn.shape type` represents a static tensor shape.
    }];
}

def TS_PartialShape : DialectType<TensorShapeDialect,
                               CPred<"$_self.isa<::cinn::dialect::PartialShapeType>()">, "!cinn.partial_shape type">,
  BuildableType<"$_builder.getType<::cinn::dialect::PartialShapeType()"> {
  let typeDescription = [{
      `!cinn.partial_shape type` represents eight a static tensor shape, unranked tensor shape or
      a ranked tensor shape with unknown dimension sizes.
    }];
}
