---
type: note
status: draft
tags:
  - computer
  - tools
  - repo
rating: 0
create: 2026-05-12
update:
---

这是一个24年9月的笔记，同步到我们新的仓库
[[TODO]] 这是现成的博客文章，还需要进一步整理

# git repo工具详细使用教程——彻底学会Android repo的使用

## repo是什么？

repo是Google开发的用于管理Android版本库的一个工具，repo是使用Python对git进行了一定的封装，并不是用于取代git，它简化了对多个Git版本库的管理。用repo管理的版本库都需要使用git命令来进行操作。**因此，使用repo工具之前，请先确保已经安装git。**

## 为什么要用repo？

项目模块化/组件化之后各模块也作为独立的 Git 仓库从主项目里剥离了出去，各模块各自管理自己的版本。Android[源码](https://so.csdn.net/so/search?q=%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020)引用了很多开源项目，每一个子项目都是一个Git仓库，每个Git仓库都有很多分支版本，为了方便统一管理各个子项目的Git仓库，需要一个上层工具批量进行处理，因此repo诞生。  
repo也会建立一个Git仓库，用来记录当前Android版本下各个子项目的Git仓库分别处于哪一个分支，这个仓库通常叫做：manifest仓库(清单库)。

## repo下载安装

下载地址：https://mirrors.tuna.tsinghua.edu.cn/git/git-repo ，将下载下来的文件命名为repo，放在PATH环境变量所包含的目录下面，例如可以放在`/usr/local/bin`目录下（后面介绍均以放在`/usr/local/bin`目录下为例）。

或者，直接使用curl命令下载：

```bash
curl https://mirrors.tuna.tsinghua.edu.cn/git/git-repo > /usr/local/bin/repo
```

最后，修改repo文件的执行权限：`chmod 777 /usr/local/bin/repo`。

> 其实下载下来的repo文件只是一个使用Python编写的引导脚本（Google 称之为 Repo launcher，本质上是一个python脚本，可以使用vim打开的），完整的repo(即，repo的主体部分)还没有下载。

## repo help

查看repo帮助说明，该帮助列举了repo所支持的子命令，及各个子命令的简要介绍。  
如果需要查看某个具体子命令的详细介绍，执行命令`repo help <command>`即可。例如查看`repo init`的帮助，可以输入`repo help init`。

> 上一小节已经提及到了，下载下来的repo只是一个引导脚本，完整的repo工具还没有下载，如下图所示，此时使用`repo help`只能看到`init`和`help`两个子命令，而且帮助信息中还会提示repo还未安装，需要执行`repo init`安装。(需要注意`repo init`需要跟参数的，后面会单独介绍`repo init`的使用)

![在这里插入图片描述](https://img-blog.csdnimg.cn/999aba99a21848348515bafa7ecbbe41.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn6a2U546L54ix5a2m5Lmg,size_20,color_FFFFFF,t_70,g_se,x_16)

> 当执行完`repo init`下载了完整的repo工具之后，再执行`repo help`就会看到repo更多的子命令。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/fd8e577a153d4687b5c11ddd104f7f07.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn6a2U546L54ix5a2m5Lmg,size_20,color_FFFFFF,t_70,g_se,x_16)

## repo version

**命令格式：**

```bash
repo version
```

查看repo的版本

## repo selfupdate

**命令格式：**

```bash
repo selfupdate
```

用于 repo 自身的更新。如果有新版本的repo存在, 这个命令会升级repo到最新版本。通常这个动作在repo sync时会自动去做, 所以不需要最终用户手动去执行。

**常用选项：**

- `--no-repo-verify`：不要验证repo源码.

## repo init

### repo init命令

**命令格式：**

```bash
repo init [options] [manifest url]
```

例如：

```bash
repo init -u manifest_git_path -m manifest_file_name -b branch_name --repo-url=repo_url --no-repo-verify
```

**命令效果：**  
首先当前目录产生一个`.repo`目录  
然后克隆一份repo的源代码到`.repo/repo`下，里面存放了其他repo子命令，即repo的主体部分。  
接着从`manifest_git_path`仓库地址clone清单库到`.repo/manifests`和`.repo/manifests.git`目录。  
同时`.repo`目录下还包括manifest仓库(清单库)内容

**常用选项：**

- `-u`：指定Manifest库的Git访问路径。**唯一必不可少的选项**
- `-m`：指定要使用的Manifest文件。不指定的话，默认为default.xml文件
- `-b`：指定要使用Manifest仓库中的某个特定分支。
- `--repo-url`：指定repo的远端repoGit库的访问路径。
- `--no-repo-verify`：指定不要验证repo源码。
- `--mirror`：创建远程存储库的副本，而不是客户端工作目录。该选项用于创建版本库镜像。使用该选项则在下一步`repo sync`同步时，本地按照源的版本库组织方式进行组织，否则会按照 manifest.xml 指定的方式重新组织并检出到本地

### 修改获取repo的源码路径

前面已经说了下载下来的repo只是一个引导脚本，当执行`repo init`的时候才会下载repo的主体部分，并存放在当前目录的`.repo/repo`目录下。

这里就会涉及到一个问题，repo的主体部分是从哪里下载的？其实查看repo的引导脚本(/usr/local/bin/repo)可以发现，repo主体部分默认从`https://gerrit.googlesource.com/git-repo`获取(即，执行`repo init`命令时，不设置`--repo-url`选项)，这个网站需要科学上网才可以访问。

解决该问题可以使用其他镜像源来获取，例如使用清华源。具体执行上有多种方式，下面列举两种方式供参考：  
**方式一：**  
每次执行repo init时，增加选项`--repo-url=https://gerrit-googlesource.lug.ustc.edu.cn/git-repo`

**方式二：（建议）**  
设置环境变量`REPO_URL`，例如：

```bash
export REPO_URL='https://mirrors.tuna.tsinghua.edu.cn/git/git-repo'
```

可以将环境变量写在启动脚本中(如，`/etc/profile`)

## .repo文件夹简介

执行`repo init`命令之后，会在当前目录创建一个`.repo`文件夹。下面看看该文件夹下面都有什么东西吧。

```bash
$ tree .repo -L 1
.repo
├── manifests
├── manifests.git
├── manifest.xml
└── repo

3 directories, 1 file
```

| 文件夹           | 描述                                                               |
| ------------- | ---------------------------------------------------------------- |
| manifests     | manifest仓库(清单库)内容，即`repo init`的`-u`选项对应的仓库                       |
| manifests.git | manifest仓库(清单库)的`.git`目录                                         |
| manifest.xml  | 指明当前生效的Manifest文件，即`repo init`的`-m`选项对应的参数(没有该选项时默认为default.xml) |
| repo          | repo 命令的主体，包含了最新的 repo 命令                                        |

## manifest文件介绍

所谓manifest仓库(清单库)其实就是存放manifest(清单)文件的仓库，实际上可以是任意仓库，只要该仓库中存在`repo init`命令`-m`选项指定的manifest文件即可，清单库命名为`manifest`只不过是一种约定俗成的写法罢了。

manifest仓库一般都会有一个default.xml文件，该文件为默认的manifest文件。

### manifest文件格式

manifest文件是用XML文件的格式记录了本项目依赖的各个Git仓库的名称、地址，以及分支等信息。

下面举个实际的例子，看下manifest文件是什么样

```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest>
    <remote fetch="ssh://git@git.software.team/learn-repo" name="origin" review="http://xxx.xxx.xxx:8080"/>
    <default remote="origin" revision="master" sync-j="4" />
    <project name="build" path="build">
        <linkfile dest="build.sh" src="build.sh"/>
    </project>
    <project name="docs" path="docs">
        <copyfile dest="README.md" src="README.md"/>
    </project>
    <project name="third_party/openssl" path="third_party/openssl" revision="OpenSSL_1_1_1l" />
    <project name="src" path="src" revision="release" />
</manifest>
```

**1. remote元素**

- `fetch`：使用此remote的所有项目的Git URL前缀。 每个项目的名称都附加到此前缀以形成用于克隆项目的实际 URL。如果使用此remote的所有项目的前缀和manifest仓库前置一致的话，可以使用`..`代替。
- `name`：此清单文件唯一的短名称。此处指定的名称用作每个项目的 `.git/config` 中的远程名称，因此可自动用于 `git fetch`、`git remote`、`git pull` 和 `git push` 等命令。
- `review`：通过`repo upload`将评论上传到的 Gerrit 服务器的主机名。 **该属性是可选的； 如果未指定，则`repo upload`将不起作用**。

**2. default元素**

- `remote`：project部分不单独指定remote的话就使用default部分的。
- `revision`：project部分不单独指定revision的话就使用default部分的。
- `sync-j`：同步时(执行`repo sync`命令时)使用的并行作业数

**3. project元素**  
该部分定义了项目代码由哪些子仓库组成

- `name`：相对于remote部分`fetch`指定的前缀的相对路径
- `path`：把代码下载下来后在本地的相对于当前路径的相对路径
- `revision`：是指下载下来的代码要checkout到哪个revision上，这里的revision可以是commit id、branch name、tag name，本质上都是commit id。default.xml中通常用branch name做revision，可以下载到并且checkout出该branch上最新的代码，标签和/或commit id在理论上应该有效，但尚未经过广泛测试。如果revision用commit id的话，那后面必须跟上upstream，upstream的值是个branch name。revision部分如果省略的话，就等于使用default部分定义的revision。

**4. copyfile元素**  
project元素的子元素，每个元素描述了一对 src-dest 文件对。同步时(即执行`repo sync`命令时)`src`文件会被拷贝到`dest`。通常会被用于 README 或 Makefile 或其他构建脚本。  
`dest`：是相对于当前目录(执行`repo init`和`repo sync`命令的目录)的路径  
`src`：是相对于project path的相对路径

**5. linkfile元素**  
与`copyfile`类似，只不过不是拷贝，而是建立软连接。

更多关于manifest文件的格式的介绍，可以在`repo init`命令成功执行后，在代码根目录下的`.repo/repo/docs`下看到相关文档说明(问：不会写manifest文件，就无法创建清单库，从而无法执行repo init怎么办？答：找个开源清单库执行`repo init`即可，或者去github上下载一份repo的源码)。最简单的方式是直接去网站[repo Manifest Format](https://gerrit-googlesource.proxy.ustclug.org/git-repo/+/HEAD/docs/manifest-format.md)查看帮助。

## repo sync

**命令格式：**

```bash
repo sync [<project>...]
```

初始化好一个 repo 工作目录后下一步就是把代码同步下来了，该命令用来下载新的更改并更新本地环境中的工作文件。如果您在未使用任何参数的情况下运行 `repo sync`，则该操作会同步所有项目(所有项目是指manifest文件中所有的project元素)的文件。

`<project>`：为manifest文件中project元素的`name`属性或者`path`属性的值。如果只需要同步某一个或者个别几个项目的话，就可以采用这种方法。

运行`repo sync` 后，将出现以下情况：

- 如果目标项目从未同步过，则 `repo sync` 相当于 `git clone`。远程代码库中的所有分支都会复制到本地项目目录中。

- 如果目标项目已同步过，则 repo sync 相当于以下命令：
  
  ```bash
  git remote update
  git rebase origin/<BRANCH>
  ```
  
  其中 `<BRANCH>` 是本地项目目录中当前已检出的分支。如果本地分支没有在跟踪远程代码库中的分支，则相应项目不会发生任何同步。

- 如果 `git rebase` 操作导致合并冲突，那么您需要使用普通 Git 命令（例如 `git rebase --continue`）来解决冲突。

repo sync 运行成功后，指定项目中的代码会与远程代码库中的代码保持同步。

**常用选项：**

- `-d`：将指定项目切换回清单修订版本。如果项目当前属于某个主题分支，但只是临时需要清单修订版本，则此选项会有所帮助。
- `-s`：同步到当前清单中清单服务器元素指定的一个已知的良好版本。
- `-f`：即使某个项目同步失败，系统也会继续同步其他项目。
- `-t`：使用对应 tag 里的 manifest 文件
- `-m`：手动指定当前操作使用哪个 manifest 文件
- `--force-sync`：如果需要，强制覆盖现有的 git 目录指向不同的对象目录。**此操作可能会导致数据丢失**

## repo start 创建主题分支

**命令格式：**

```bash
repo start <newbranchname> [--all | <project>...]
```

创建并切换分支。**刚克隆下来的代码是没有分支的**，`repo start`实际是对`git checkout -b`命令的封装。  
为指定的项目或所有的项目（若使用`-all`），以清单文件中为设定的分支，创建特定的分支。

**常用选项：**

- `<newbranchname>` 参数应简要说明您尝试对项目进行的更改。
- `<project>` 指定了将参与此主题分支的项目。

> 注意：`.`是一个非常实用的简写形式，用来代表当前工作目录中的项目。

这条指令与`git checkout -b` 还是有很大区别的。

- `git checkout -b` 是在当前所在的分支的基础上创建特性分支。
- `repo start` 是在清单文件设定的分支的基础上创建特性分支。

## repo status

**命令格式：**

```bash
repo status [<project>...]
```

查看文件状态。对于每个指定的项目，将工作树与临时区域（索引）以及此分支 (HEAD) 上的最近一次提交进行比较。在这三种状态存在差异之处显示每个文件的摘要行。

要仅查看当前分支的状态，请运行 `repo status`。系统会按项目列出状态信息。对于项目中的每个文件，系统使用两个字母的代码来表示：

- 在第一列中，大写字母表示临时区域与上次提交状态之间的不同之处。
  
  | 字母  | 含义    | 描述                        |
  | --- | ----- | ------------------------- |
  | -   | 无更改   | HEAD 与索引中相同               |
  | A   | 已添加   | 不存在于 HEAD 中，但存在于索引中       |
  | M   | 已修改   | 存在于 HEAD 中，但索引中的文件已修改     |
  | D   | 已删除   | 存在于 HEAD 中，但不存在于索引中       |
  | R   | 已重命名  | 不存在于 HEAD 中，但索引中的文件的路径已更改 |
  | C   | 已复制   | 不存在于 HEAD 中，已从索引中的另一个文件复制 |
  | T   | 模式已更改 | HEAD 与索引中的内容相同，但模式已更改     |
  | U   | 未合并   | HEAD 与索引中的内容相同，但模式已更改     |

- 在第二列中，小写字母表示工作目录与索引之间的不同之处。
  
  | 字母  | 含义   | 描述                    |
  | --- | ---- | --------------------- |
  | -   | 新/未知 | HEAD 与索引中相同           |
  | m   | 已修改  | 存在于索引中，也存在于工作树中（但已修改） |
  | d   | 已删除  | 存在于索引中，不存在于工作树中       |

两个表示状态的字母后面，显示文件名信息。如果有文件重名还会显示改变前后的文件名及文件的相似度。

## repo checkout

**命令格式：**

```bash
repo checkout <branchname> [<project>...]
```

切换分支。 实际上是对git checkout命令的封装，但不能带`-b`参数，所以不能用此命令来创建特性分支。  
该命令等同于：`repo forall [<project>...] -c git checkout <branchname>`

## repo branch

该命令等同于`repo branches`  
**命令格式：**

```bash
repo branches [<project>...]
```

汇总当前所有可用的主题分支。

## repo diff

**命令格式：**

```bash
repo diff [<project>...]
```

查看工作区文件差异。实际是对`git diff`命令的封装，用于分别显示各个项目或指定项目工作区下的文件差异。在 commit 和工作目录之间使用 `git diff` 显示明显差异的更改。

## repo stage

**命令格式：**

```bash
repo stage -i [<project>...]
```

把文件添加到index表中。实际上是对`git add --interactive`命令的封装，用于挑选各个项目中的改动以加入暂存区。

**常用选项：**

- `-i`：表示`git add --interactive`命令中的`--interactive`，给出一个界面供用户选择。

## repo forall

```bash
repo forall [<project>...] -c <command> [<arg>...]
```

在每个项目中运行指定的 shell 命令。通过 `repo forall` 可使用下列额外的环境变量：

- `REPO_PROJECT`：项目的名称。
- `REPO_PATH`：项目在该工作区的相对路径。
- `REPO_REMOTE`：项目远程仓库的名称。
- `REPO_LREV`：manifest文件中revision属性，已转换为本地跟踪分支。如果您需要将manifest中revision值传递到某个本地运行的 Git 命令，则可使用此变量。
- `REPO_RREV`：manifest文件中revision属性，与manifest文件中显示的名称完全一致。

**常用选项：**

- `-c`：要运行的命令和参数，即shell命令。此命令会通过 /bin/sh 进行求值，它之后的任何参数都将作为 shell 位置参数传递。
- `-p`：在指定命令输出结果之前显示项目标头。这通过以下方式实现：将管道绑定到命令的 stdin、stdout 和 sterr 流，然后通过管道将所有输出结果传输到一个页面调度会话中显示的连续流中。
- `-v`：显示该命令向 stderr 写入的消息。

> 注意：shell指令中有上述环境变量时，则需要用使用单引号把shell命令括起来。

**示例：**

1. 打印项目列表
   
   ```bash
   repo forall -c 'echo $REPO_PROJECT'
   ```

2. 打印项目路径
   
   ```bash
   repo forall -c 'echo $REPO_PATH'
   ```

## repo prune

**命令格式：**

```bash
repo download {[project] change[/patchset]}...
```

删除已经合并分支。实际上是对`git branch -d`命令的封装，该命令用于扫描项目的各个分支，并删除已经合并的分支。

## repo abandon

**命令格式：**

```bash
repo abandon [--all | <branchname>] [<project>...]
```

删除指定分支。实际是对`git brance -D`命令的封装。

## repo upload

**命令格式：**

```bash
repo upload [--re --cc] [<project>]...
```

对于指定的项目，Repo 会将本地分支与最后一次 repo sync 时更新的远程分支进行比较。Repo 会提示您选择一个或多个尚未上传以供审核的分支。

> **注意：使用`repo upload`需要搭建gerrit环境，并且在manifest文件`remote`元素中添加`review`属性**

您选择一个或多个分支后，所选分支上的所有提交都会通过 HTTPS 连接传输到 Gerrit。您需要配置一个 HTTPS 密码以启用上传授权。要生成新的用户名/密码对以用于 HTTPS 传输，请访问[密码生成器](https://android-review.googlesource.com/new-password)。

当 Gerrit 通过其服务器接收对象数据时，它会将每项提交转变成一项更改，以便审核者可以单独针对每项提交给出意见。要将几项“检查点”提交合并为一项提交，请使用 `git rebase -i`，然后再运行 `repo upload`。

如果您在未使用任何参数的情况下运行 `repo upload`，则该操作会搜索所有项目中的更改以进行上传。

要在更改上传之后对其进行修改，您应该使用 `git rebase -i` 或 `git commit --amend` 等工具更新您的本地提交。修改完成之后，请执行以下操作：

- 进行核对以确保更新后的分支是当前已检出的分支。

- 对于相应系列中的每项提交，请在方括号内输入 Gerrit 更改 ID：
  
  ```bash
  # Replacing from branch foo
  [ 3021 ] 35f2596c Refactor part of GetUploadableBranches to lookup one specific...
  [ 2829 ] ec18b4ba Update proto client to support patch set replacments
  # Insert change numbers in the brackets to add a new patch set.
  # To create a new change record, leave the brackets empty.
  ```

上传完成后，这些更改将拥有一个额外的补丁程序集。

`repo upload` 相当于 `git push`，但是又有很大的不同。它将版本库改动推送到代码审核服务器（Gerrit软件架设）的特殊引用上。代码审核服务器会对推送的提交进行特殊处理，将新的提交显示为一个待审核的修改集，并进入代码审核流程，只有当审核通过后，才会合并到官方正式的版本库中。

**常用选项：**

- `-t`：发送本地分支名称到Gerrit代码审核服务器
- `--re=REVIEWERS`：要求指定的人员进行审核
- `--cc=CC`：同时发送通知到如下邮件地址

## repo download

**命令格式：**

```bash
repo download {[project] change[/patchset]}...
```

从审核系统中下载指定更改，并放在您项目的本地工作目录中供使用。  
例如，要将[更改 23823](https://android-review.googlesource.com/23823) 下载到您的平台/编译目录，请运行以下命令：

```bash
$ repo download platform/build 23823
```

`repo sync` 应该可以有效移除通过 `repo download` 检索到的任何提交。或者，您可以将远程分支检出，例如 `git checkout m/master`。

> `repo download`命令主要用于代码审核者下载和评估贡献者提交的修订。  
> 贡献者的修订在Git版本库中`refs/changes//引用方式`命名（缺省的patchset为1），和其他Git引用一样，用`git fetch`获取，该引用所指向的最新的提交就是贡献者待审核的修订。  
> 使用`repo download`命令实际上就是用`git fetch`获取到对应项目的`refs/changes//patchset>`引用，并自动切换到对应的引用上。

## repo grep

**命令格式：**

```bash
repo grep {pattern | -e pattern} [<project>...]
```

打印出符合某个模式的行。相当于对 `git grep` 的封装，用于在项目文件中进行内容查找。

## repo manifest

**命令格式：**

```bash
repo manifest [-o {-|NAME.xml}] [-m MANIFEST.xml] [-r]
```

manifest检验工具，用于显示当前使用的manifest文件内容。

**常用选项：**

- `-r, --revision-as-HEAD`：把某版次存为当前的HEAD
- `-o -|NAME.xml, --output-file=-|NAME.xml`：把manifest存为NAME.xml

**示例：**

```bash
# 获取仓库的sha1值，并记录在一个新的release.xml文件中
repo manifest -o release.xml -r
```

## repo 工作流程

常用的 repo 工作流程如下：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/9cf8e032fec246d685bcdb0e09e7d445.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn6a2U546L54ix5a2m5Lmg,size_19,color_FFFFFF,t_70,g_se,x_16)

## repo sync后 git branch -a 显示中 “no branch” 和 “remotes/m/master” 的含义

使用repo工具同步代码之后，进入任意项目路径下，执行 `git branch -a` 输出如下：

```bash
$ git branch -a
* (no branch)
  remotes/m/master -> origin/dev
  remotes/origin/dev
  remotes/origin/master
```

1. 输出中的后两行比较好理解，就是该仓库的远程仓库里存在的分支。

2. 输出的第一行 `* (no branch)` 中的 `*` 表示当前所在的分支，该行意思是，当前不再任何分支上。
   
   **为什么会显示`no branch` 呢？**  
   `repo sync` 只是根据清单文件中 `revision` 版本进行更新的，没有固定的branch，`repo sync` 成功之后，不能直接进行操作，需要先执行 `repo start` 建立新的分支进行开发。
   
   其实，执行 `repo branches` 命令也会显示 `no branches` 的，这个就更好理解了，不同的子仓库的 `revision` 不尽相同，所有git仓库放在一起，更是没有一个确切的branch了。

3. 输出的第二行中 `remotes/m/master -> origin/dev` 又是什么意思呢？
   
   - 前一部分表示 repo 清单库(manifest仓库)的分支，即执行 `repo init` 命令时 `-b` 选项的参数，如果执行 `repo init` 命令时没有指定 `-b` 选项，则表示默认采用清单库的`master`分支。
   - 箭头所指的后一部分 `origin/dev`，表示当前所使用的清单文件(.xml文件)里面指定的单个git库的revision值（即，清单文件中 `project` 元素的 `revision` 属性，如果 `project` 元素没有指定 `revision` 属性，则默认使用的是 `default` 元素的 `revision` 属性）。
   
   **这样做的目的是：为了让用户方便的知道自己目前工作在清单库的哪个分支上。当前的清单库的这个分支又引用了当前git库的哪个branch/tag上**
