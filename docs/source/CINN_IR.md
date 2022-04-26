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

