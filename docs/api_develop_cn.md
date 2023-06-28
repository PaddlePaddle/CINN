# CINN 算子 API 开发指南

CINN作为国产化基础科技的开源攻坚项目，力图攻克深度学习编译器相关问题，我们社区非常欢迎大家加入和进行开源代码贡献。我们将在本文档讲述在CINN代码中如何开发一个新的算子API

## 背景知识

### 熟悉CINN IR的DSL
CINN IR是CINN底层进行计算表达的IR（Intermediate Representation），这些 IR 可以构成DSL（Domain Specific Language），关于CINN IR的抽象语法树见TODO（插入链接），关于CINN IR DSL在C++的基本介绍和写法见CINN/tutorials/matmul.cc

### CINN 算子在框架里的结构
