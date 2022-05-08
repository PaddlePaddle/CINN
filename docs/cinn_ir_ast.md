# CINN IR AST

```
Expr = _Module_(std::string name, Expr* buffers, Expr* functions, Expr* submodules)
     | _LoweredFunc_(std::string name,
                   Argument* args,
                   Buffer* temp_bufs,
                   Expr body,
                   Expr* alloc_output_buffer_exprs,
                   Expr* dealloc_output_buffer_exprs,
                   Expr* buffer_data_cast_exprs,
                   Expr* argument_prepare_exprs
                   )
     | _Var_(std::string name, Expr lower_bound, Expr upper_bound)  
     | Block(Expr* stmts)

     -- unit of schedule IR which represents tensor computation
     | ScheduleBlock(Var* iter_vars,
                     Expr* read_buffers,
                     Expr* write_buffers,
                     std::string name,
                     Expr body)
     -- execute time ScheduleBlock with binding iter_values
     | ScheduleBlockRealize(Expr* iter_values, Expr schedule_block)
 
     -- operator Expr
     | Cast(Type type, Expr operand)
     | Let(Expr symbol, Expr body)
     | UnaryOpNode(Type type, Expr* operands) -- reuse ExprNode operands, so it is a vector
     | BinaryOpNode(Type type, Expr* operands)
     | Reduce(Expr init, Expr body, utils::SmallVector<Var, 4> reduce_axis, ReduceType reduce_type)
     | Broadcast(Expr value, int lanes)
     | Load(Expr tensor, Expr* indices)
     | Store(Expr tensor, Expr value, Expr* indices)
     | Alloc(Expr destination, Expr* extents, Expr? condition, Expr? body)
     | Free(Expr destination)
    
     -- control flow Expr
     | Select(Expr condition, Expr true_value, Expr false_value)
     | IfThenElse(Expr condition, Expr true_case, Expr false_case)
     | ForBase
     | Call(std::string name,
            Expr* read_args,
            Expr* write_args,
            CallType call_type,
            FunctionRef func,
            int value_index,    -- The output value index if func value is a tuple
            std::map<std::string, absl::variant<int, float, bool, std::string>> attrs
            )

     -- basic variable or data types
     | Ramp(Expr base, Expr stride, int lanes)
     | IntImm(Type t, int64_t value)
     | UIntImm(Type t, int64_t value)
     | FloatImm(Type t, double value)
     | StringImm(Type t, std::string value)

UnaryOpNode = -- numerical calculation op
              Minus
              -- logical op
            | Not

BinaryOpNode = 
             -- numerical calculation op
               Add | Sub | Mul | Div | Mod 
               
             -- max/min
             | Max | Min
 
             -- comparator op
             | EQ | NE | LT | LE | GT | GE

             -- logical op
             | And | Or | Not -- NOTE: not support Xor now
             
             -- function ops
             | FracOp(Expr n, Expr d)
             | PowerOp(Expr n, Expr d)
             | Product(Expr* vs)
             | Sum(Expr* vs)
             

ForBase = For(ForType for_type, Var loop_var, Expr min, Expr extent, Expr body)
        | PolyFor(ForType for_type, Var iterator, Expr init, Expr condition, Expr inc, Expr body)

-- enum style classes:

Type = F16 ｜ F32 | F64 | I1 | I8 | I16 | I32 | I64 
     | UI1 | UI8 | UI16 | UI32 | UI64 
     | Void | String 
     | Customized
     | vector(Type, int lanes)
     | const(Type type)
     | handle(Type type) -- pointer type, such as `int*`
     | handlehandle(Type type) -- pointer of pointer, such as `int**`.

CallType = Extern | CINN | Intrinsic | ISL

ReduceType = Sum | Sub | Mul | Div | Max | Min

ForType = Default | Serial | Vectorized | Unrolled | GPUThread ｜ GPUBlock ｜ GPULane
```
