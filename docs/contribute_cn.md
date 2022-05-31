# 贡献指南

CINN作为国产化基础科技的开源攻坚项目，力图攻克深度学习编译器相关问题，我们社区非常欢迎大家加入和进行开源代码贡献。我们将在本文档讲述CINN代码贡献的基本软件提交指南


## Git 操作和 CINN 编译测试指南

CINN 使用 Git 作为基本版本控制工具，Github作为发布开源代码的基础平台。以下教程讲指导读者您如何使用 Git 给CINN提交代码

### Fork CINN repo

使用自己的账号 `USERNAME` 登陆Github，到我们的 [CINN: https://github.com/PaddlePaddle/CINN](https://github.com/PaddlePaddle/CINN) GitHub 首页，然后单击 `Fork` 按钮，`Fork` 会对开源仓库（repo）产生一个您自己的复制，生成自己目录下的仓库，比如 <https://github.com/USERNAME/CINN> 

###  克隆（Clone）

将自己的远程仓库 clone 到本地：

```bash
git clone https://github.com/USERNAME/CINN
cd CINN
```

这个是远程代码仓库在本地的克隆仓库，可以在这个仓库进行本地开发，以及发送代码更改到远程仓库

### 在本地使用新分支开发功能
我们在这一节只介绍日常开发常用的分支（branch）管理，不涉及发版、修复等。CINN完整的分支管理规范与PaddlePaddle一致，具体请参考 [Paddle 分支规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/others/releasing_process.md)。

从日常开发简单步骤出发，所有的 feature 或 bug fix 的开发工作都应该在您自己的一个新的分支上完成，CINN 以 `develop` 分支作为自己最新代码所在分支，一般从 `develop` 分支上创建新分支。

以下代码使用 `git checkout -b` 创建并切换到新分支 `test`。您也可以使用其他新分支名字。

```bash
git checkout -b test
```

### 使用Docker

Docker [http://www.docker.com](http://www.docker.com) 是一种把软件和依赖环境都打包发布的技术，它使得开发人员减少配置一个软件的依赖软件和环境。我们推荐您使用Docker管理和运行CINN，方便处理第三方依赖。

使用以下命令拉取运行 CINN 的 docker 镜像：
```bash
docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
```

Docker 镜像相当于对一个运行环境和软件的打包存档。我们用以下命令，创建并启动 docker 容器。Docker 容器是在您机器上进行运行的环境，其拥有自己的文件系统、进程管理等：
```bash
docker run -it -v $PWD:/CINN /registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82 --name=cinn_test bin/bash
```

注意以上的-v命令将本地的文件夹映射到 docker 容器中，--name指定了启动容器的名字，您可以根据自己的需要，更改相关参数。

常用的 docker 命令还有：
```bash
exit                           # Docker里运行，退出当前容器
docker stop cinn-test          # 停止名为cinn-test的容器
docker start cinn-test         # 启动名为cinn-test的容器
docker exec -it cinn-test bash # 进入一个已经启动的名为cinn-test的容器
docker rm cinn-test            # 删除名为cinn-test的容器
docker rmi cinn-image          # 删除名为cinn-image的镜像
```

### 本地修改CINN代码
经过上面步骤，您可以在本地您自己的分支修改 CINN 代码了。在这个步骤，有2个常见 git 命令帮助您跟踪自己的代码修改，我们首先介绍 `git status`，这会提示当前目录的一些变化，比如您修改了`README.md` ，添加了一个 `test.txt` 文件，在 CINN 本地仓库输入`git status` ，会显示：

```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

同时您可以通过
```bash
git diff README.md
```
来查看`README.md`您具体和上一个 git 提交(commit)之间修改了什么

### 编译CINN和本地检查测试

#### 自动编译和运行CINN当下测试的方法
修改完代码，可以编译和测试 CINN，CINN 在仓库根目录提供了一个脚本 `build.sh` 可以帮助您自动编译和测试。以下是测试内容和相应命令。

1. CPU(X86) 后端测试: `bash ./build.sh ci`
2. NVGPU(cuda) + CUDNN 后端测试: `bash ./build.sh gpu_on ci`
3. NVGPU(cuda) 后端关闭 CUDNN 测试: `bash ./build.sh gpu_on cudnn_off ci`

#### 手动编译和运行测试的方法
您也可以手动编译和安装 `CINN`，运行自己的代码，我们以都打开 cuda ，cudnn ，并且编译相关测试为例：
```bash
mkdir build
cd build
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_TESTING=ON
make -j
```

以上 cmake 选项可以根据用户需要的编译选项调整。更多选项见 CMakelists.txt 文件

在make完成后，可以通过 `ctest` 命令手动运行想要运行的 CINN 测试

#### 编译结束后生成 pip whl 包
不论手动还是自动编译，结束后会生成 `build/python/dist/cinn-xxxxx.whl` ，具体文件名会因为平台、选项、依赖而不同。这个 whl 文件可以通过 pip 安装，之后您就可以使用。
```bash
pip install build/python/dist/cinn-xxxxx.whl
```

### 提交代码更改
对于您本地修改好代码后，需要通过 `git add` 命令把想要正式修改的文件添加，然后用 `git commit` 命令创建一个提交（commit）, commit 包含了一部分代码更改，是分支上的基本更改单位。

例如要添加对 `README.md` 的修改，使用
```bash
git add README.md
```

另一个常用命令是
```bash
git add -a
```
该命令添加本地 repo 中所有的代码修改

之后，运行
```bash
git commit
```
会弹出一个窗口，让您输入对该次 `commit` 的基本文字表述。请用简洁的语言完整描述您的代码进行的修改。

这里有几点需要注意的地方：

1. 由于 Github 空间限制，只提交代码相关更改，对于深度模型数据集，或其他大文件，不应该提交为 git commit 。
2. 对于涉及您账号密码、本地网关、个人隐私、数据版权等网络安全相关的内容，请勿提交到 Github，因为 Github 相关更改会开源掉，且更改分支过程会留下记录，涉及网络安全的内容公开掉，可能会引起您的财产损失等。

### 同步远程更改到本地

在您的开发过程中，因为现代软件工程往往不止一个人开发，不同人可能同时修改同一处的代码，或者您需要获取别人在其他分支的最新代码，您需要在本地同步远程 repo 的更改。

首先通过 `git remote` 查看当前远程仓库的名字。

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/CINN (fetch)
origin	https://github.com/USERNAME/CINN (push)
```

这里 origin 是我们 clone 的远程仓库的名字，也就是自己用户名下的 CINN，接下来我们也把原始 CINN 仓库的远程添加进来管理，命名为 upstream。

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/CINN
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

### 把本地修改 commit 推送到远程仓库
使用 `git push` 命令将本地的修改推送到 GitHub 上您 fork 的 repo 的 test 分支，也就是 https://github.com/USERNAME/CINN。

```bash
➜  git push origin test
```

这样，您可以登陆您的 Github 上的 CINN repo，看到您的代码修改在这里有所显示了。

### 提交 PR

PR 全称为 Pull Request，是从您的 repo 向项目 repo ，在本例子中也就是CINN repo，提交相应更改请求的基本单位。

我们建议 PR 为一个完整的小功能，它不要代码太多，比如小于500行，以免未来检查代码不方便

在完成上节的 `git push` 后，您的代码修改在自己 repo 下有一个新分支，在 Github 的分支选择切换到所建分支，然后点击 `Compare & pull request`。或者也可以通过点击 `Contribute` 然后点击 `Open pull request`

选择PR的目标分支，在 CINN 项目往往是 `develop` 分支。如果是对release 修改的 cherry-pick，则使用相应 release 分支

在弹出来的界面填写好 PR 标题和 PR描述

### CINN repo 单元测试

你在Pull Request中每提交一次新的commit后，会触发CI单元测试，显示在您PR界面的底下，它一般会在几个小时内完成。

通过的测试出现了绿色的勾，表示你本次commit通过了各项单元测试，你只需要关注显示Required任务，不显示的可能是我们正在测试的任务。

如果所需的测试后出现了红色叉号，代表你本次的commit未通过某项单元测试，在这种情况下，请你点击detail查看报错详情log，优先自行解决报错问题，进行代码修改，提交新commit会重新触发测试。

由于CI测试为百度飞桨团队管理维护，外部人员想要重新运行，或者别的相关操作，可能没有权限。这时可以通过issue、线下对接等形式联系我们，我们的工作人员将和你一起查看。

### PR review 和合入
提交PR后，可以等待，也可以直接联系百度CINN工作人员，催促进行代码review。当下代码需要百度CINN工作人员review通过，并操作进行PR merge。待PR merge后，您的代码更改就同步到了CINN的repo，您可以在CINN官方repo看到相应更改。

### 删除分支（可选）

如果你比较有代码洁癖，提交后的代码分支不想本地保存，可以用以下命令删除本地分支。

```bash
# 切换到 develop 分支
➜  git checkout develop

# 删除 my-cool-stuff 分支，因为您无法在一个分支上删除其本身
➜  git branch -D my-cool-stuff
```

在Github网站上您自己repo的分支可以通过网站按键删除。

## 代码风格

本小节介绍CINN开发使用的代码风格，请尽量参考以下规范，进行代码开发：

- C++：[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Python：[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
