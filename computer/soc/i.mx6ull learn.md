---
type: note
status: draft
tags:
  - computer
  - soc
rating: 0
create: 2026-05-12
update:
---
## 常用linux命令

### 磁盘

fdisk -l

### samba

添加密码：sudo smbpasswd -a username

### 权限

Linux chmod（英文全拼：change mode）命令是控制用户对文件的权限的命令

Linux/Unix 的文件调用权限分为三级 : 文件所有者（Owner）、用户组（Group）、其它用户（Other Users）。

![img](attachments/file-permissions-rwx.jpg)

只有文件所有者和超级用户可以修改文件或目录的权限。可以使用绝对模式（八进制数字模式），符号模式指定文件的权限。

![img](attachments/rwx-standard-unix-permission-bits.png)

**使用权限** : 所有使用者

```
chmod [-cfvR] [--help] [--version] mode file...
```

**查看文件权限**

```
ls -l [fliename]
```

### 解压

```
# 压缩文件 file1 和目录 dir2 到 test.tar.gz
tar -zcvf test.tar.gz file1 dir2
# 解压 test.tar.gz（将 c 换成 x 即可）
tar -zxvf test.tar.gz
# 列出压缩文件的内容
tar -ztvf test.tar.gz

.tar.gz     格式解压为          tar   -zxvf   xx.tar.gz

.tar.bz2   格式解压为          tar   -jxvf    xx.tar.bz2
```





## MAKEFILE

### 学习笔记

编译步骤：*.s => *.o =>  * .elf  => *.bin

```makefile
objs := start.o main.o # objs变量 先链接start.o 再链接 main.o

ledc.bin:$(objs) # 通过objs 生成最终目标 ledc.bin
	arm-linux-gnueabihf-ld -Timx6ul.lds -o ledc.elf $^			# 使用imx6ul.lds链接脚本指定的地址，（由$^, 指当前的变量，即objs）生成 ledc.elf
	arm-linux-gnueabihf-objcopy -O binary -S ledc.elf $@		# 由ledc.elf生成目标bin文件（$@,即ledc.bin）
	arm-linux-gnueabihf-objdump -D -m arm ledc.elf > ledc.dis	# 反汇编
	
%.o:%.s
	arm-linux-gnueabihf-gcc -Wall -nostdlib -c -O2 -o $@ $<		# 具体文件的编译命令，$@ 目标文件（即%.o）， $<第一个依赖文件（即%.s）
	
%.o:%.S
	arm-linux-gnueabihf-gcc -Wall -nostdlib -c -O2 -o $@ $<
	
%.o:%.c
	arm-linux-gnueabihf-gcc -Wall -nostdlib -c -O2 -o $@ $<
	
clean:
	rm -rf *.o ledc.bin ledc.elf ledc.dis
```

### makefile介绍

```
一 Makefile一般的格式是：
	target:components 
		rule
		
二 $@、$^、$<
这三个分别表示：
$@          --代表目标文件(target)
$^          --代表所有的依赖文件(components)
$<          --代表第一个依赖文件(components中最左边的那个)。
$?          --代表示比目标还要新的依赖文件列表。以空格分隔。
$%          --仅当目标是函数库文件中，表示规则中的目标成员名。例如，如果一个目标是"foo.a(bar.o)"，那么，"$%"就是"bar.o"，"$@"就是"foo.a"。如果目标不是函数库文件（Unix下是[.a]，Windows下是[.lib]），那么，其值为空。

'-' 符号的使用
通常删除，创建文件如果碰到文件不存在或者已经创建，那么希望忽略掉这个错误，继续执行，就可以在命令前面添加-
     -rm dir；
     -mkdir aaadir；
 
$ 主要扩展打开makefile中定义的变量
$$ 符号主要扩展打开makefile中定义的shell变量

反斜杠（\）是换行符的意思

```

### makefile 是如何工作的

1. 在默认的方式下，也就是只输入make命令。 make会在当前目录下找名字叫“Makefile”或“makefile”的文件。

 	2. 如果找到，它会找文件中的第一个目标文件（target），并把这个文件作为最终的目标文件。
 	3.   如果target文件不存在，或是target所依赖的后面的 .o 文件的文件修改时间要比edit这个文件新，那么，他就会执行后面所定义的命令来生成target这个文件。
 	4.   如果target所依赖的.o文件也存在，那么make会在当前文件中找目标为.o文件的依赖性，如果找到则再根据那一个规则生成.o文件。（这有点像一个堆栈的过程）
 	5.   C文件和H文件是存在的啦，于是make会生成 .o 文件，然后再用 .o 文件声明make的终极任务，也就是执行文件target了。

推荐链接：[Makefile教程（绝对经典，所有问题看这一篇足够了）-CSDN博客](https://blog.csdn.net/weixin_38391755/article/details/80380786)



### 一个好用的make模板

这个 makefile 的巧妙之处在于它能自动为你确定依赖，你所要做的就是把你的 C/C++ 文件放到 `src/` 文件夹中。

```makefile
# Thanks to Job Vranish (https://spin.atomicobject.com/2016/08/26/makefile-c-projects/)
TARGET_EXEC := final_program

BUILD_DIR := ./build
SRC_DIRS := ./src

# Find all the C and C++ files we want to compile
# Note the single quotes around the * expressions. Make will incorrectly expand these otherwise.
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')

# String substitution for every C/C++ file.
# As an example, hello.cpp turns into ./build/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
    $(CC) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
    mkdir -p $(dir $@)
    $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
    mkdir -p $(dir $@)
    $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
    rm -r $(BUILD_DIR)

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)
```

## 链接脚本

```
SECTIONS{
    . = 0X10000000;		代码链接到该位置
    .text : {*(.text)}	定义 text 段
    . = 0X30000000;		text段的启示位置
    .data ALIGN(4) : { *(.data) } 	定义data段， ALIGN(4)：4字节对齐
    .bss ALIGN(4) : { *(.bss) } 	定义bss(已定义但未初始化)
}
```



## vim 操作

**如何复制黏贴：** 

1 命令行按 v 进入visual mode

2 shift + 方向键选中 按 y 复制 d 剪切

3 按p / P 黏贴

4 退出 visual mode

5 撤销： 命令行  :u

6  要使用 系统粘贴板 的内容，也可以直接在命令模式按 Shift + Inset 进行粘贴

```
yy    复制游标所在行整行 
2yy或y2y    复制 2 行
y^    复制至行首，或y0
y$    复制至行尾
yw    复制一个word
y2w    复制两个word 
yG    复制至文件尾
y1G    复制至文件首

dd    剪切游标所在行整行 
d^    剪切至行首，或d0
d$    剪切至行尾 
dw    剪切一个word 
dG    剪切至文件尾

p    粘贴至游标后（下） 
P    粘贴至游标前（上）
```



## uboot

网络

```
=> setenv ipaddr 192.168.1.50
=> setenv ethaddr b8:ae:1d:01:00:00
=> setenv gatewayip 192.168.1.1
=> setenv netmask 255.255.255.0
=> setenv serverip 192.168.1.108
=> saveenv

ping 192.168.1.108

dhcp // 获取ip

//nfs 下载固件

=> nfs 80800000 192.168.1.108:/home/alientek/linux/nfs/zImage
Using FEC1 device
File transfer via NFS from server 192.168.1.108; our IP address is 192.168.1.109
Filename '/home/alientek/linux/nfs/zImage'.
Load address: 0x80800000
```

启动等待时间

```
CONFIG_BOOTDELAY
setenv bootdelay 2
saveenv
```

**通过网络启动**

```
setenv bootargs 'console=ttymxc0,115200 root=/dev/nfs nfsroot=192.168.1.109:/home/alientek/linux/nfs/lvtao_rootfs,proto=tcp rw ip=192.168.1.190:192.168.1.109:192.168.1.1:255.255.255.0::eth0:off'

setenv bootcmd 'tftp 80800000 zImage; tftp 83000000 imx6ull-alientek-emmc.dtb; bootz 80800000 - 83000000'

saveenv

printenv
```



```
设置环境变量

=> setenv bootargs 'console=ttymxc0,115200 root=/dev/mmcblk1p2 rootwait rw'
=> setenv bootcmd 'tftp 80800000 zImage; tftp 83000000 imx6ull-alientek-emmc.dtb; bootz 80800000 - 83000000'
=> saveenv

直接下载文件
tftp 80800000 zImage
tftp 83000000 imx6ull-alientek-emmc.dtb
bootz 80800000 – 83000000

// 一个问题记录：
Wrong Ramdisk Image Format
Ramdisk image is corrupt or invalid
原因是 bootz 命令有问题，中间的短线前后各有一个空格

```

## linux移植

![image-20240704002431085](attachments/image-20240704002431085.png)



运行 imx6ull_alientek_emmc.sh

注意开启网络驱动

```
> Device Drivers
    -> Network device support
        -> PHY Device support and infrastructure
            -> Drivers for SMSC PHYs
```



## linux驱动

**查看CPU频率**

```
cat /sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state
```

**查看驱动设备树对应节点是否有了**

```
/sys/firmware/devicetree/base # ls	# 看设备树配的对不对
cat /proc/interrupts	# 看中断是不是有了
ls /dev		# 看驱动是不是挂上了
```



**挂载与卸载**

```
depmod  		第一次加载时执行
modprobe *.ko  	加载
rmmod *.ko 		卸载
```



对于Linux驱动开发，C++是一种常用的编程[语言](https://geek.csdn.net/educolumn/05c8de7581a582e6d46821757663ed30?spm=1055.2569.3001.10083)。在Linux内核中，C++的使用受到了一些限制，例如不能使用STL和异常处理等特性。但是，C++仍然可以用于编写Linux驱动程序，特别是对于一些需要面向[对象](https://geek.csdn.net/educolumn/04c51611e4b730957464192e0307b82c?spm=1055.2569.3001.10083)编程的场景。

在Linux驱动开发中，C++可以使用一些特殊的技术来实现面向[对象](https://geek.csdn.net/educolumn/04c51611e4b730957464192e0307b82c?spm=1055.2569.3001.10083)编程。例如，可以使用C++的类来封装设备驱动程序，并使用虚[函数](https://geek.csdn.net/educolumn/ba94496e6cfa8630df5d047358ad9719?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NDQ0MDg2MiwiZXhwIjoxNzA3MzcxOTM4LCJpYXQiOjE3MDY3NjcxMzgsInVzZXJuYW1lIjoid2VpeGluXzY4NjQ1NjQ1In0.RrTYEnMNYPC7AQdoij4SBb0kKEgHoyvF-bZOG2eGQvc&spm=1055.2569.3001.10083)来实现多态性。此外，C++还可以使用RAII技术来管理资源，避免内存泄漏等问题。

**需要注意的是，在Linux内核中使用C++需要遵循一些规则和约定。例如，需要使用特殊的编译选项来编译C++，并且需要避免使用一些C++特性，如RTTI和异常处理等。**

### Unknown symbol device_create (err -22)

**加载驱动时出现以下报错**

```

/lib/modules/4.1.15 # depmod
/lib/modules/4.1.15 # modprobe led.ko
led: disagrees about version of symbol device_create
led: Unknown symbol device_create (err -22)
led: disagrees about version of symbol device_destroy
led: Unknown symbol device_destroy (err -22)
led: disagrees about version of symbol device_create
led: Unknown symbol device_create (err -22)
led: disagrees about version of symbol device_destroy
led: Unknown symbol device_destroy (err -22)
modprobe: can't load module led.ko (led.ko): Invalid argument
```

**解决办法** 编译驱动的源码与编译内核的源码不一致，1、更新内核。或者 2 修改编译驱动用的内核路径.。

### 设备树

编译命令

```
make dtbs
```

### I2C 驱动

I2C 设备接线

```
对于mpu6050 及 096oled_i2c
add 接 GND
VCC 接 3.3
sda 接开发板 42
scl 接开发板 43
```



**i2cdetect**

```
/lib/modules/4.1.15 # i2cdetect -a 0
i2cdetect: WARNING! This program can confuse your I2C bus
Continue? [y/N] y
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00: 00 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c -- -- --
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- UU -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


## i2cdetect -l  //列出系统中存在的 I2C 总线列表，只有一条I2C0
##  i2cdetect -y 0    // 对I2C0 总线探测总线上的i2c设备
返回参数说明：
‘--’：表示该地址被检测，但没有芯片应答；
‘UU’：表示该地址当前由内核驱动程序使用；
‘**’：**表示以十六进制表示的设备地址编号。

 ## i2cdump -f -y 0 0x68 //读取 I2C 总线 0 上地址为 0x68 的设备寄存器内容
 ##  i2cset -f -y 0 0x68 0x06 0x18  //设置为18年
 ##  i2cget -f -y 0 0x68 0x06       //读地址0x06的寄存器值
```

### MPU6050 驱动

开发板接法 SCL->43 SDA->42 3.3  供电

![image-20240714220619184](attachments/image-20240714220619184.png)

![image-20240714220715061](attachments/image-20240714220715061.png)

```
VCC：电源正
GND：
SCL：I2C串行时钟线 or SPI 串行时钟端口
SDA：I2C串行数据线 or SPI 串行数据输入
XDA: 连接其他I2C设备的主机数据口
XCL: 给I2C设备提供主时钟
AD0: I2C器件地址选择 or SPI串行数据输出
INT: 中断引脚

I2C 地址：
#define MPU_WRITE   0XD0//MPU6050的AD0接低电平
#define MPU_READ    0XD1//MPU6050的AD0接低电平
```





### 0.96 寸OLED驱动

中景园 0.96寸OLED有两种驱动模式，可配置显示器的外围电路切换。点阵大小为127*64，通过控制内部寄存器bit位位0/1控制对应点亮暗

#### I2C 模式驱动

![image-20240714182020869](attachments/image-20240714182020869.png)

I2C模式下的驱动较为简单，就当普通I2C设备进行读写就行了，需要注意的是I2C地址，**IMX6ULL 的地址只有七位，0x78实际变成了0x3c (看起来是最后一位删掉了)**，使用 i2cdetect 能够扫到对应的I2C地址。接线跟设备树都很简单。

代码上主要实现如何发送命令和发送数据。

```
#define OLED_CMD  0x00  //OLED写命令
#define OLED_DATA 0x40  //OLED写数据

// 写入单字节命令或数据
static s32 oled_write_byte(u8 reg, u8 para)
{
    u8 data[2];
    struct i2c_msg msg;
    //此处将.probe函数中所保存的私有数据强制转换为i2c_client结构体
    struct i2c_client *client = (struct i2c_client *)oled.private_data;
    data[0] = reg;              //寄存器
    data[1] = para;             //参数
    msg.addr = client->addr;    //设备树中的地址, 0x3c
    msg.flags = 0;              //标记为写
    msg.buf = data;             //要写入的数据缓冲区
    msg.len = 2;                //要写入的数据长度
    return i2c_transfer(client->adapter, &msg, 1);
}

// 写入多字节数据，注意这里的实现必须第一字节为 0x40
// data[0] == OLED_DATA
static s32 oled_write_byte_array(u8* data, u8 len)
{
    struct i2c_msg msg;
    struct i2c_client *client = (struct i2c_client *)oled.private_data;
    msg.addr = client->addr;        //设备树中的地址, 0x3c
    msg.flags = 0;                  //标记为写
    msg.buf = data;                 //要写入的数据缓冲区
    msg.len = len;                  //要写入的数据长度
    return i2c_transfer(client->adapter, &msg, 1); 
}

// 屏幕显示的例子（清屏）
void oled_clear(void)
{  
    u8 i;
    u8 data[129] = {};
    memset(data, 0, 129);
    data[0] = OLED_DATA;
    for(i=0; i<8; i++) 
    {  
        oled_write_byte(OLED_CMD, 0xb0 + i);    // 设置页地址（0~7）
        oled_write_byte(OLED_CMD, 0x00);        // 设置显示位置—列低地址
        oled_write_byte(OLED_CMD, 0x10);        // 设置显示位置—列高地址
        oled_write_byte_array(data, 129);		// 写入数据
    }
}

// 一些命令，具体可以看数据手册, 相邻两行没写注释的与上面的组成双局命令
SendByte(0xAE);	//关闭显示
SendByte(0xD5); //设置显示时钟分频比/振荡器频率
SendByte(0x80);
SendByte(0xA8); //设置多路复用率
SendByte(0x3F);
SendByte(0xD3); //设置显示偏移
SendByte(0x00);
SendByte(0x40); //设置显示开始行
SendByte(0xA1); //设置左右方向，0xA1正常 0xA0左右反置
SendByte(0xC8); //设置上下方向，0xC8正常 0xC0上下反置
SendByte(0xDA); //设置COM引脚硬件配置
SendByte(0x12);
SendByte(0x81); //设置对比度控制
SendByte(0xCF);
SendByte(0xD9); //设置预充电周期
SendByte(0xF1);
SendByte(0xDB); //设置VCOMH取消选择级别
SendByte(0x30);
SendByte(0xA4); //设置整个显示打开/关闭
SendByte(0xA6); //设置正常/倒转显示
SendByte(0x8D); //设置充电泵
SendByte(0x14);
SendByte(0xAF); //开启显示


```

**寻址模式**

寻址模式有三种，水平寻址模式，垂直寻址模式，页寻址模式（最常用）
又因为水平寻址方式和垂直寻址方式在写入整个屏幕时方便，水平寻址模式和页寻址模式的取模方法相同
所以这里介绍水平寻址模式和页寻址模式

| 模式         | 命令 |
| ------------ | ---- |
| 水平寻址模式 | 0x00 |
| 垂直寻址模式 | 0x01 |
| 页寻址模式   | 0x02 |

```c
SendByte(0x00);	//进入命令设置模式
SendByte(0x20);	//进入寻址设置模式
SendByte(0x02);	//选择寻址模式
123
```

**页寻址**

OLED屏幕就是一个个小的有机自发光二极管组成的阵列，0.96OLED屏幕的分辨率是128*64，即每行有128个发光二极管，一共有64行。页是芯片设计者为了方便将**同一列的8个点阵编成一组，用一个8bit数表示，这样的8行128个数被称为1页**
![请添加图片描述](attachments/b1f2db04adac4081aafe2f3fd8209c99.png)
用0和1来代表灭和亮（阳码)，越靠近**上方**的是低位

![在这里插入图片描述](attachments/c0f7853b338d4100ac3069255cf755ec.png)

> 左边的实际的点阵，右边是等效的数，这样编码之后的这页这列的值为0x05

![请添加图片描述](attachments/43abae557c314f328fe590f0a10ca859.png)

> 经过编码之后，128*64的OLED屏幕有**8页**，128行

**水平寻址**

核心特点为：横向编码，列地址自动加，遇到设置范围的页尾时自动跳转到下一页，传输到设置范围的最后一页最后一列时自动复位



![请添加图片描述](attachments/ea53da2dfd3a4afa8ec22b01f7df3628.png)

![在这里插入图片描述](attachments/88dcd2b0b70d49d2a21f91f424a4cf82.png)

#### SPI 模式驱动

[嵌入式linux之iMX6ULL驱动开发 | 通用spi驱动之spidev使用总结_linux spidev-CSDN博客](https://blog.csdn.net/yyz_1987/article/details/131918983)

SPI 版屏幕的接口定义如下，默认使用的**四线SPI模式** 注意所有引脚都要使用，只有一个SPI时，地址选择可以接地

![image-20240714214647686](attachments/image-20240714214647686.png)

具体接法参考下面的：

![image-20240714214926158](attachments/image-20240714214926158.png)

**测试接线SPI3**

参考代码：https://blog.csdn.net/AURORA1997/article/details/119335398

SPI3定义如下：I.MX6ULL芯片资料/IMX6ULL参考手册.pdf P193

![image-20240714215159912](attachments/image-20240714215159912.png)

另外额外使用了(GPIO1_IO31, GPIO1_IO24)两个引脚作为D/C 和 res,设备树如下。

开发板接法：

cs->17 dc->44 res->33 D1->20 D0->18, 3.3 供电



**注意要确定引脚别被其他设备占用，有占用后不能正常工作**(注意下面的配置与触摸的I2C引脚冲突)

```xml
CS--17--UART2--TXD--SPI3--SS0--GPIO1-io20
DC--44--UART5--RXD--GPIO1--IO31
D1--20--UART2_CTS--SPI3--MOSI
D0--18--UART2_RXD--SPI3--SCLK

pinctrl_ecspi3: ecspi3grp {
    fsl,pins = <
        MX6UL_PAD_UART2_TX_DATA__GPIO1_IO20     0x100b0  /* CS*/
        MX6UL_PAD_UART2_RTS_B__ECSPI3_MISO      0x100b1  /* MISO*/
        MX6UL_PAD_UART2_CTS_B__ECSPI3_MOSI	    0x10B1   /* MOSI*/
        MX6UL_PAD_UART2_RX_DATA__ECSPI3_SCLK	0x10B1   /* CLK*/
    >;
};
pinctrl_oled96_reset: oled96resetgrp {
    fsl,pins = <
    	/* used for oled96 reset */
    	MX6UL_PAD_UART3_TX_DATA__GPIO1_IO24           0x10B0
    >;
};
pinctrl_oled96_DC: oled96DCgrp {
    fsl,pins = <
        /* used for oled96 DC */
        MX6UL_PAD_UART5_RX_DATA__GPIO1_IO31	    0x10B0
    >;
};

// 注意 驱动里面查找结点时应该是：
// oled96dev.nd = of_find_node_by_path("/soc/aips-bus@02000000/spba-bus@02000000/ecspi@02010000");
// 实测写道根节点下好像会报错，不确定
&ecspi3 {
    fsl,spi-num-chipselects = <1>;
    cs-gpio = <&gpio1 20 GPIO_ACTIVE_LOW>;
    reset-gpio = <&gpio1 24 GPIO_ACTIVE_LOW>;
	dc-gpio = <&gpio1 31 GPIO_ACTIVE_LOW>;
    pinctrl-names = "default";
    pinctrl-0 = <&pinctrl_ecspi3
                 &pinctrl_oled96_reset
				 &pinctrl_oled96_DC>;
    status = "okay";
    spidev: oled@0{
        compatible = "ldysl, spi_oled096";
        spi-max-frequency = <8000000>;
        reg = <0>;
    };
};
```

### 4.3寸 LCD（800*400）及触摸驱动

测试的屏幕支持24bit RGB888，及RGB565 （实测24bit 16 bit都能正常显示）

```
// LCD 驱动，可以正常点亮的配置
&lcdif {
	pinctrl-names = "default";
	pinctrl-0 = <&pinctrl_lcdif_dat
		     &pinctrl_lcdif_ctrl>;
	display = <&display0>;
	status = "okay";

	display0: display {
		bits-per-pixel = <24>;
		bus-width = <24>;

		display-timings {
			native-mode = <&timing0>;
			timing0: timing0 {
			clock-frequency = <31000000>;
			hactive = <800>;
			vactive = <480>;
			hfront-porch = <40>;
			hback-porch = <88>;
			hsync-len = <48>;
			vback-porch = <32>;
			vfront-porch = <13>;
			vsync-len = <3>;

			hsync-active = <0>;
			vsync-active = <0>;
			de-active = <1>;
			pixelclk-active = <0>;
			};
		};
	};
};
```

触摸 芯片为GT9147, 支持最多5点触控（正点原子驱动实现为单点触摸）

```
触摸一个使用四个引脚，IIC,一个reset,一个中断输入
gt9147: gt9146@14 {
                compatible = "goodix,gt9147", "goodix,gt9xx";
                reg = <0x14>;
                pinctrl-names = "default";
                pinctrl-0 = <&pinctrl_tsc>;
                interrupt-parent = <&gpio1>;
                interrupts = <9 0>;
                reset-gpios = <&gpio5 9 GPIO_ACTIVE_LOW>;
                interrupt-gpios = <&gpio1 9 GPIO_ACTIVE_LOW>;
                status = "okay";
	};
两个io pinctrl
		pinctrl_tsc: tscgrp {
			fsl,pins = <
				MX6UL_PAD_GPIO1_IO01__GPIO1_IO01	0xb0
				MX6UL_PAD_GPIO1_IO02__GPIO1_IO02	0xb0
				/*MX6UL_PAD_GPIO1_IO03__GPIO1_IO03	0xb0*/
				MX6UL_PAD_GPIO1_IO04__GPIO1_IO04	0xb0

                /* 4.3 寸 RGB 屏幕,GT9147 */
                MX6UL_PAD_SNVS_TAMPER9__GPIO5_IO09 0x10B0 /* TSC_RST */
                MX6UL_PAD_GPIO1_IO09__GPIO1_IO09 0x10B0 /* TSC_INT */
			>;
		};
		
注意需要处理下IO冲突，例如，这里发现触摸的IO与SPI olde屏幕的io有一个冲突
实际使用建议搜索：gpio9 5, 以及i2c使用的io
```

tslib 测试的时候注意要先加载驱动，否则会出现如下报错：

```
tslib ts_setup: No such file or directory
```

### input 设备与按键（KEY）驱动

```
命令记录：

dump 输入内容：
hexdump /dev/input/event*

查看input设备信息：
cat /proc/bus/input/devices
```



## Linux 应用编程

### lcd 显示

问题记录：

1、ioctl 获取lcd 信息失败，原因：fd传错（copy csdn的垃圾代码）

2、lcd 屏幕左上角有个黑点：

```
现象：
1、左上角有黑点，刷屏后黑点马上出现（手机录视频可以看到先是正常的颜色，然后马上变成黑色）
2、打印黑点位置的内存发现，写入后马上检查值是对的，过一会就编程0了
```

原因：黑色缺口为linux控制台 光标，找到：

Device Drivers > Graphics support > Console display driver support, 关闭后即正常

### 库移植坑点记录

1、zlib库移植时无能通过 .config 指定编译器，此时编出来的时x86,需要指定CC编译器：

```
export CC=arm-linux-gnueabihf-gcc
```

2、libjpeg 库编好测试时如果运行失败（打印乱码啥的），原因也可能是编译成了x86，使用 file 命令查看，就知道是啥格式了

```
file xxx.so
file xxx.lib
```

### cjson

```
cJSON.h:58:24: error: expected ‘;’ at end of member declaration void *(*malloc_fn)(size_t sz);

原因没有size_t 的定义，size_t 在 string.h 中，添加该头文件即可
```



## QT开发

注意：正常情况下触摸相当于鼠标，如果没有触摸

1、检查/dev/input/event* ,看看哪个是触摸的event，然后修改 /etc/profile 中 ts_lib 中相关配置。

2、检查触摸的驱动是否加载（当前加载驱动的脚本在 /lib/modules/4.1.15/test.sh）, 同时在/etc/init.d/rcS中运行了该脚本



## 坑点记录

1、更新设备树 用reboot重启可能不行，可以断电试试。

2、设备树的兼容跟程序里面的最好一字不差（空格什么的都一样，直接复制最好）



## 记录以备后续调试

1、当前开发板是从SD卡加载bootloader, bootloader 设置从网络启动（win 虚拟机）。emmc中的镜像已经损坏，无法启动。

2、如何设置uboot 网络启动，需要参考该文档 uboot->通过网络启动部分，以及开发板教程中过于uboot配置网络启动章节。具体需要加载的位置 看 /home/alientek/lvtao/Linux_Drivers/1_chrdevbase/readme.md 有记录

3、key 按键已经搞成回车，屏幕10分钟息屏，按key可以点亮（版上驱动：/lib/modules/4.1.15）.触摸驱动 单点驱动

4、重要代码位置（虚拟机）：

​	驱动测试app: /home/alientek/lvtao/Linux_Drivers/1_chrdevbase 支持git。具体参看内部readme

​	qt测试：/home/alientek/lvtao/qt_test

​	linux 源码（zImage 设备树修改。 注意修改config 前需要备份，不要轻易clean）,脚本：imx6ull_alientek_emmc.sh 可以编译、copy产物

​	开发板已经移植：freetype-2.8 jpeg-9b libpng-1.65 tslib-1.21 zlib-1.2.10。相关代码：/home/alientek/lvtao/Linux_Drivers/ 输出结果移动到 /home/alientek/lvtao/Linux_Drivers/tools 下了。每个第三方库目录下都有一个 lvtao_build.sh 用于编译。但你需要检查下输出目录是否正确。

**虚拟机密码 8位数字**

虚拟机名字 Ubuntu 64位

账号：Alientek 

samba 设置密码 sudo smbpasswd -a alientek


















