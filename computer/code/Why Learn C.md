---
type: book
status: done
tags:
  - c
rating: 0
create: 2026-09-21
update: 2026-09-22
---
本书是一本C语言偏进阶的书。除了介绍C语言基本知识外，还结合工程实践给出了很多例子。
全书共分为三部分：
1. 第一部分介绍C语言的基础语法
2. 第二部分是介绍了一些高阶的内容（很多日常开发不一定用的上）
3. 第三部分介绍了一些例子，与其他教材不同，这里的例子更偏向生产代码，因此更值得一看。

以下是书中总结的一些细节：
# 第一部分
1. 什么时候要避免移动数据：It’s best to avoid moving larger objects (size ≥16 bytes)
2. 对与复杂的const 语句，从右向左读：reading them right-to-left helps,
3. 对于指针类型的const：but for pointer declarations, the benefit is that **const is always east of what it’s making const**（east 即左侧）
4. **多用double**：you should generally use double by default unless you have storage constraints in which case you should use float
5. C语言中能够创建新类型的只有：enumerations (§7), structures (§10), or unions (§11).
6. 限制使用固定长度类型：The fixed-width types should be used only if if you need to represent a value contained within a specific number of bits
	1. 表示字节大小时用 size_t
	2. The only times you typically need a fixed-size integer is when you “serialize” a value,
7. 自动类型转换原则（完整见原书）
	1. int 提升（其他类型会升级为int，或者uint）:An operand of char, short, bit-field (§10.7), all either signed or unsigned, or enumeration (§7), is promoted to int if int can represent all the values of the original type or unsigned int if not.
	2. 范围允许时，无符号数转换为有符号数：If the signed type can represent all the value of the unsigned type, the operand with the unsigned type is converted to the signed type.
	3. int准换为float：Integer types are converted to floating-point types when necessary. • Smaller types are converted to larger types when necessary.
8. C 语言中一个表达式中求值顺序没有定义（除了&& 有短路机制外），特别注意函数传递参数时：fun(++i, i+j, --j)，这样的表达式是未定义的
9. 数组下标可以为负数（end[-1]），这里利用了数组本质是指针的特性
10. 忽略表达式求值时可以将返回值强转为void，不强转编译器会报warning
11. sizeof 可以不带括号（有区别）：If you don’t want to have to remember when to use () with sizeof, you can simply always use ().
12. 运算符的结合性，拿不准就多加括号
13. 数字的字面量默认是int，其他类型需要加后准、制定unsigned 等
14. 单个字符 字面量也是int：Oddly, character literals are of type int and not char as you might expect.
15. 文件内作用域static：Unless you want to make a file-scope variable “public,” declare it static.
16. register 修饰（建议编译器，不一定采纳）：Only parameters and local variables may be declared register. Note that it’s only a hint that the compiler is free to ignore.
17. 不要用 register：You therefore shouldn’t use register in any new C code.
18. typedef：为啥C++用decltype，原因是C++有引用类型
	1. If you know C++ , typeof is like its decltype. Why isn’t typeof called decltype in C also? (Or why isn’t decltype in C++ called typeof?) The short answer is that C++ has references and C doesn’t — and that affects the type deduced.
19. typedef_unqual：removes all top-level qualifiers
20. in C, any non-zero value is considered true
21. do-while 必须要有"{}"：Unlike either while or for, do-while is invariably used with {} even when there is only a single statement
22. goto 仅以下场景适用：
	1. When going to shared error-handling code as first shown in listing
	2. When similarly going to shared cleanup code.
	3. When inside a nested loop, to break out of all loops.
	4. To break out of a loop from within a switch
23. label 的作用域
	1. Labels are local to functions
	2. Label names are in their own namespace, so it’s perfectly legal to have a label and a variable with the same name in the same function
24. Use goto either to go to shared code or to break out of a loop when break is insufficient.
25. void指针不能解引用：A void* can never be dereferenced since void objects don’t exist
26. array 与指针：
	1. 没办法知道指针指向一个变量还是数组。C 把这个职责留给编译器了：There is no way to tell if a pointer points to a single object or multiple objects.
	2. array 变量也不知道自己多长（函数传参数时 array 退化为指针）：Since the name of an array decays into a pointer in expressions, there is no way to tell how many elements an array has either
	3. 多维数组推荐指针数组方式实现：using an array of pointers to strings is simply more convenient.
27. 变长数组（VLA）：
	1. 栈存储，太大会造成栈溢出，编译器也无法预测是否安全：While VLAs work and seem convenient, one serious caveat to VLAs is that, if the length is too big, it will silently overflow the stack resulting in undefined behavior (§15). Additionally, unlike malloc returning nullptr upon failure, there’s no way to detect when a VLA overflows the stack. Hence, if the size can be “too big,” your code has to guard against it
	2. C++ never adopted VLAs from C.
	3. **Don’t use VLAs**
28. 枚举：
	1. 没有作用域，因此要手动制定前缀避免冲突（C++ 枚举类加了作用域）enumeration constants aren’t scoped, which means they’re all “injected” into the surrounding scope
	2. 可以指定枚举的原始类型：enum color : unsigned char
	3. 可以多个不同枚举对应一个变量值，但switch中不允许（switch 利用跳转表）
	4. switch + 枚举时，最好不要加 default。这样漏掉的case 编译器会提示，加了default，漏掉就走default了
	5. 枚举常用于 bitwise flags（可以用｜或 & 来运算）
29. 预处理器：
	1. 预处理器不懂C语言语法：Other than the set of tokens that comprise C (identifiers, operators, punctuation, literals, comments), the preprocessor knows very little about C. In particular, it knows nothing about scope
	2. 编译命令可以定义宏： -D command-line option can be used to define a macro having an optional value
	3. \#define 定义宏时：If value is omitted, it defaults to 1.
	4. **defined：** The only use for defined is if you want to distinguish between a macro not being defined from being defined to false:
	5. the normal C rules do not apply to the preprocessor.
	6. 头文件
		1. The <> form looks for the requested file from one of a set of pre-defined paths (the include path) and is largely used to include files from the C standard library or third-party libraries.
		2. The ”” form looks for the requested file starting relative to the current directory first, then the include path, and is used to include files from your program.
	7. 宏要用括号括起来，避免展开后异常：When multiple statements comprise a macro, it’s best to enclose them within a dowhile loop (§5.5) as shown in listing 8.3.
	8. do-while 把多个语句括起来（避免跟其他语句连起来）。Using a do-while will keep multiple statements grouped together as a compound statement and execute exactly once.
	9. \#embed \<path\> Embeds path from a set of pre-defined system paths. # embed ”path” Embeds path relative to the current directory.
	10. \#pragma : Pragmas are a way to issue compiler-specific commands.
	11. \#pragma once (现代编译器自动加，已经不需要了): Modern preprocessors implicitly do what \#pragma once does with ordinary include guards, so \#pragma once isn’t really needed any more.
30. 形参 formal parameters、实参 actual parameters
	1. array 类型形参：Another quirky feature of C is that the use of [] for function parameters is just syntactic sugar since the compiler rewrites such parameters as pointers
	2. array parameters don’t exist in C.
	3. array 形参数组长度可以固定：void print_csi_10( int const vals[static 10] );
31. inline：
	1. inline 失败条件：
		1. 太大
		2. 通过函数指针调用
		3. 递归调用
		4. The function has a loop
	2. an inline function must be defined in a header file.
32. 结构体
	1. 结构体不能嵌套定义：As far as the compiler is concerned, it’s as if coord were declared at file scope。Even so, declaring nested structures in C is confusing — don’t do it.
	2. The compiler would generate assembly instructions that includes shifts and a mask
	3. 位域结构体：Structure members can be bit-fields, but don’t use them unless there’s a specific reason for doing so.
33. IO:
	1. By default, stdin is fd 0, stdout is fd 1, and stderr is fd 2.
	2. 读写文件时系统调用开销更大（陷入内核），因此调用标准库的开销更小（有buffer）That’s why fread and fwrite do buffering: to minimize system calls
	3. Hence, you can use FILE* most of the time, and file descriptors only when you have to.
	4. 内存文件：Opening a file creates an in-memory link to it that counts exactly the same as a directory link.
	5. 读取环境变量：Environment variables can be read by your programs as another way to input information in addition to either command-line options or files.
34. 程序组织：
	1. including local headers before standard headers.
	2. 推荐名称：module_init and module_cleanup
35. 多线程（C 的多线程库目前不完善，建议用系统库）：
	1. **detached**: Alternatively, a thread can be detached meaning that when it terminates, its **resources will be cleaned-up automatically**.
	2. volatile 不是加锁，不保证多线程安全（其他语言有这样用的）The use of volatile to attempt to make something thread-safe is always wrong in C
	3. 递归锁：A recursive mutex is one that allows it to be locked more than once by the same thread.
	4. 条件变量虚假唤醒：Due to intricacies in some thread implementations, it’s possible that cnd_wait can receive a “spurious” signal and wake up even when no work is available.
	5. call_once（并不是没有开销，实际编译器会生成标识位，每次检查）
	6. 线程局部存储
# 第二部分
1. 未定义行为：The compiler is allowed to assume that undefined behavior never happens.
	1. 一个例子是（高优化级别）：函数内第一行直接解引用，也就告诉编译器p不可能为空（实际可能写错），编译器认为后续的if不可能执行，就全优化了。
	```
	fun(int* p){
		int a = *p;
		if(p==nullptr)
		{
		....
		}
	}
	```

2. Atomic（C 的原子变量）
	1. memory_order_consume: In contrast, a strongly-ordered CPU (a strong memory model) is one where every core sees values change in memory in the same order that another core wrote them.
	2. 对齐确保结构体在同一个cache行：ensures that head and tail reside on different cache lines and eliminates false sharing between them.
3. Common Bugs：建议看原文
	1. 数组越界
	2. buffer overflow
	3. double free
	4. null pointer dereference
	5. off by one (C string)
	6. use after free
	7. memory leak
	8. uninitialized variable
4. setjmp longjmp：日常开发基本不用，可用于实现异常（最后一张有介绍）
	However, C does have two functions, setjmp and longjmp, that can be used together to return through more than one level of the call stack, typically as an error-handling mechanism when there’s no other way for a deeply nested function to stop and return an error.
5. volatile:
	1. The use of volatile prevents the compiler from storing the value of a variable in a register (among other things), hence it’s stored in the local stack frame instead and so is unaffected by register value restoration.
	2. 三个需要volatile的地方：
		1. 告诉编译器，修饰的变量可能会变化（编译器不知道），因此不要优化。例如有些变量值实际是外部硬件传进来的，单片机、linux驱动等开发常见。
		2. For use in a signal handler.
		3. setjmp 要用
6. restrict 告诉编译器传入的多个指针不会重叠（独立的指针、指向的内存也独立），这样可以生成更快速的命令（少一些汇编指令）。最常见的地方就是 memcpy 函数
* 高亮, page 309
# 第三部分
这里主要介绍了五个例子：
1. 带buf、长度的字符串，类似C++ vector 实现
2. list
3. hash map
4. dynamic dispatch：
	1. 利用指针数组实现类似C++ 多态（虚函数、虚表），类内存储，占空间且必须继承父类（引入不想要的东西）
	2. fat_pointer 函数数组独立全局存储，对象跟虚函数指针绑定一起作为“胖指针”。go rust 等实现。优点是：1. 函数指针不占类空间，2不用继承就能获得想要的函数。缺点是：指针定义改变了，不能向C++ 那样用裸指针同时有多态
5. 介绍用 longjmp/setjmp 实现的一个 try、catch、finally 结构的异常。可以作为C++异常的原型。深入了解异常机制值得仔细研究。


