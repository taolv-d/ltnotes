这是一个24年9月的笔记，同步到我们新的仓库

git reset 回退分支

git reset HEAD~1

**–mixed**  
**不删除工作空间改动代码，撤销commit，并且撤销git add . 操作**  
这个为默认参数,git reset --mixed HEAD^ 和 git reset HEAD^ 效果是一样的。

**–soft**  
不删除工作空间改动代码，撤销commit，不撤销git add .

**–hard**  
删除工作空间改动代码，撤销commit，撤销git add .  
注意完成这个操作后，就恢复到了上一次的commit状态。

git revert 撤销 某次操作，此次操作之前和之后的commit和history都会保留，并且把这次撤销作为一次最新的提交

**git revert 和 git reset的区别**

1. git revert是用一次新的commit来回滚之前的commit，git reset是直接删除指定的commit。
2. 在回滚这一操作上看，效果差不多。但是在日后继续merge以前的老版本时有区别。因为git revert是用一次逆向的commit“中和”之前的提交，因此日后合并老的branch时，导致这部分改变不会再次出现，但是git reset是之间把某些commit在某个branch上删除，因而和老的branch再次merge时，这些被回滚的commit应该还会被引入。
3. git reset 是把HEAD向后移动了一下，而git revert是HEAD继续前进，只是新的commit的内容和要revert的内容正好相反，能够抵消要被revert的内容。

**git restore 和 git restore --staged的区别**

git restore指令使得在工作空间但是不在暂存区的文件撤销更改(内容恢复到没修改之前的状态)

git restore --staged的作用是将暂存区的文件从暂存区撤出，但不会更改文件的内容，即本地工作区中文件被修改内容还在。
