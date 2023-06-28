# CINN IR

CINN 的中间表达（IR）主要class 包含Expr()、IRSchedule、ScheduleHelper。

Expr 是CINN的low-level IR，它是一种AST（抽象语法树），一个Expr代表一个计算Kernel。
IRSchedule类包含了所有的schedule 原语，每一个原语是IRSchedule类的成员函数。schedule原语通过ScheduleHelper来实现，它的功能是对Expr的AST进行一系列的变换操作。

ModuleExpr表示一个模型的Expr，通过传入一个expr vector来构造，并且持有这个expr vector作为私有变量。

- 目前IRSchedule类的schedule原语有：
  
###Schedule 原语
- GetLoops (const Expr& block) 
功能：获取ModuleExpr里指导Block的全部loop

- GetAllBlocks()
功能：获取所有存在ModuleExpr的blocks
- GetBlock(const std::string& block_name)
根据name获取指定Block
- Split(const Expr& loop, const std::vector<int>& factors)
根据factor切割loop为多个loop
- Fuse
融合多个loop为1个
...

####Expr 是如何生成的？

Expr 是由高层IR- HLIR通过lower得到的。HLIR是基于图的IR，Expr是基于AST的IR。

LowerVec()函数输入为一个stage变量和一些Tensor变量，stage 包含了tensor的所有schedule，从而通过creategraph函数构建出图，最后输出lowerfunc，而lowerfunc的body成员即为每一个kernel的ast表达。

具体的调用过程如下：

LowerVec()内调用LowerImpl(), 构造出图以及初始化schedule。
再通过调用LowerImpl实例（重载括号函数）创建出func_body，其类型为std::vector<Expr>。
具体调用函数名为：GenerateFunctionBody（）-> std::vector < Expr >

GenerateFunctionBody函数调用分析:
输入为schedule
主要逻辑为遍历schedule里的groups和groups的nodes来生成vector< Expr >

####CINN IR的主要元素以及概念。

- IR的表达，代表一个值或者返回一个值。
- Stmt，IR的状态
- Tensor（输入或者临时Tensor）
- Buffer（内存buffer)

### Tensor 
Tensor 代表输入或者临时变量。每一个Tensor被分配了一个buffer。

### schedule
最原始的基于tensor的计算构造了一个SSA 图， 每一个Tensor 被赋值一个Stage，代表最基本的schedule 元素。
每一个stage 有一个 domain(isl.Set) 和一个 schedule(isl.Map), 所有的Schedule 在他们上面执行。

### Schedule the stages
CINN 借鉴了 Tiramisu项目里的思想，会遍历依赖关系图，将图分割为多个groups。
以下为最朴素的切割图的规则:
-  对于初始化，为每个阶段创建一个唯一的组（只需要id），以拓扑顺序遍历计算图和检查具有依赖关系的两条语句是否具有相同的迭代空间和域。如果两个语句被 `compute_at` 标记，也合并到同一个组，这个过程就好比一个并查集。对于每个组，使用不同的 `ast_build` 来生成 ISL IR（以便我们可以单独设置迭代器）

### Scheduler module

Scheduler 以stages为输入，做前面提到的graph分割，最后输出几个schedule 元素。

每个调度元素都拥有一个（ISL）迭代域和一个（ISL）调度，可以将其传递给 ast_gen 并生成代码。

