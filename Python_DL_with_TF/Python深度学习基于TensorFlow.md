# 第一部分 Python及应用数学基础

## 第1章 NumPy常用操作

NumPy(Numerical Python)，它提供了两种基本的对象：ndarray(N-dimensional array object)和ufunc(universal function object)。ndarray是存储单一数据类型的多维数组，而ufunc则是能够对数组进行处理的函数。

NumPy的主要特点：

* ndarray，快速，节省空间的多维数组，提供数组化的算术运算和高级的广播功能。
* 使用标准数学函数对整个数组的数据进行快速运算，而不需要编写循环。
* 读取/写入磁盘上的阵列数据和操作存储器映像文件的工具。
* 线性代数，随机数生成，以及傅里叶变换的能力。
* 集成C、C++、Fortan代码的工具。

在使用NumPy之前，需要先导入该模块：

```shell
import numpy as np
```

### 1.1 生成ndarray的几种方式 

NumPy封装了一个新的数据类型ndarray，一个多维数组对象，该对象封装了许多常用的数学运算函数。生成ndarray的几种方式，如从已有数据中创建；利用random创建；创建特殊多维数组；使用arange函数等。

#### 1.从已有数据中创建

直接对python的基础数据类型（如列表、元组等）进行转换来生成ndarray。

(1)将列表转换成ndarray

```shell
import numpy as np
list1 = [3.14, 2.17, 0, 1, 2]
nd1 = np.array(list1)
print(nd1)
print(type(nd1))
```

```shell
[3.14 2.17 0.   1.   2.  ]
<class 'numpy.ndarray'>
```

(2)嵌套列表可以转换为多维ndarray

```shell
import numpy as np
list2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd2 = np.array(list2)
print(nd2)
print(type(nd2))
```

```shell
[[3.14 2.17 0.   1.   2.  ]
 [1.   2.   3.   4.   5.  ]]
<class 'numpy.ndarray'>
```

如果把(1)和(2)中的列表换成元组也同样适合。

#### 2.利用random模块生成ndarray

在深度学习中，我们经常需要对一些变量进行初始化，适当的初始化能提高模型的性能。通常我们用随机数生成模块random来生成，当然random模块又分为多种函数：random生成0到1之间的随机数；uniform生成均匀分布随机数；randn生成标准正态的随机数；normal生成正态分布；shuffle随机打乱顺序；seed设置随机数种子等。

```shell
import numpy as np

nd5 = np.random.random([3, 3])
print(nd5)
print(type(nd5))
```

```shell
[[0.77284096 0.58126235 0.89996499]
 [0.37052975 0.20651262 0.31865204]
 [0.36542842 0.70181133 0.18882855]]
<class 'numpy.ndarray'>
```

生成一个随机种子，对生成的随机数打乱。

```shell
import numpy as np

np.random.seed(123)
nd5_1 = np.random.randn(2, 3)
print(nd5_1)
np.random.shuffle(nd5_1)
print("随机打乱后数据")
print(nd5_1)
print(type(nd5_1))
```

```shell
[[-1.0856306   0.99734545  0.2829785 ]
 [-1.50629471 -0.57860025  1.65143654]]
随机打乱后数据
[[-1.50629471 -0.57860025  1.65143654]
 [-1.0856306   0.99734545  0.2829785 ]]
<class 'numpy.ndarray'>
```

#### 3.创建特定形状的多维数组

数据初始化时，有时需要生成一些特殊矩阵，如0或1的数组或矩阵，这是我们可以利用np.zeros、np.ones、np.diag来实现。

```shell
import numpy as np

#生成全是0的3*3矩阵
nd6 = np.zeros([3, 3])
print(nd6)
#生成全是1的3*3矩阵
nd7 = np.ones([3, 3])
print(nd7)
#生成3阶的单位矩阵
nd8 = np.eye(3)
print(nd8)
#生成3阶对角矩阵
print(np.diag([1, 2, 3]))
```

```shell
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[1 0 0]
 [0 2 0]
 [0 0 3]]
```

我们还可以把生成的数据保存到磁盘，然后从磁盘读取。

```shell
import numpy as np
nd9 = np.random.random([5, 5])
np.savetxt(X = nd9, fname = './test2.txt')
nd10 = np.loadtxt('./test2.txt')
print(nd10)
```

```shell
[[0.02798196 0.17390652 0.15408224 0.07708648 0.8898657 ]
 [0.7503787  0.69340324 0.51176338 0.46426806 0.56843069]
 [0.30254945 0.49730879 0.68326291 0.91669867 0.10892895]
 [0.49549179 0.23283593 0.43686066 0.75154299 0.48089213]
 [0.79772841 0.28270293 0.43341824 0.00975735 0.34079598]]
```

#### 4.利用arange函数

arange是numpy模块中的函数，其格式为：arange([start,] stop[,step],dtype=None)。根据start与stop指定的范围，以及step设定的步长，生成一个ndarray，其中start默认为0，步长step可指定为小数。

```shell
import numpy as np

print(np.arange(10))
print(np.arange(0, 10))
print(np.arange(1, 4, 0.5))
print(np.arange(9, -1, -1))
```

```shel
[0 1 2 3 4 5 6 7 8 9]
[0 1 2 3 4 5 6 7 8 9]
[1.  1.5 2.  2.5 3.  3.5]
[9 8 7 6 5 4 3 2 1 0]
```

### 1.2 存取元素

数据生成后，有以下几种读取数据的方法。

```shell
import numpy as np

np.random.seed(2018)
nd11 = np.random.random([10])
print(nd11)
#获取指定位置的数据，获取第4个元素
print(nd11[3])
#截取一段数据
print(nd11[3:6])
#截取固定间隔数据
print(nd11[1:6:2])
#倒序取数
print(nd11[::-2])
#截取一个多维数组的一个区域内数据
nd12 = np.arange(25).reshape([5, 5])
print(nd12)
print(nd12[1:3, 1:3])
#截取一个多维数组中，数值在一个值域之内的数据
print(nd12[(nd12>3)&(nd12<10)])
#截取多维数组中，指定的行，如读取第2,3行
print(nd12[[1,2]])
#截取多维数组中，指定的列，如读取第2,3列
print(nd12[:, 1:3])
```

```shell
[0.88234931 0.10432774 0.90700933 0.3063989  0.44640887 0.58998539
 0.8371111  0.69780061 0.80280284 0.10721508]
0.3063988986063515
[0.3063989  0.44640887 0.58998539]
[0.10432774 0.3063989  0.58998539]
[0.10721508 0.69780061 0.58998539 0.3063989  0.10432774]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
[[ 6  7]
 [11 12]]
[4 5 6 7 8 9]
[[ 5  6  7  8  9]
 [10 11 12 13 14]]
[[ 1  2]
 [ 6  7]
 [11 12]
 [16 17]
 [21 22]]
```

获取数组中的部分元素除通过指定索引标签外，还可以使用一些函数来实现。

```shell
import numpy as np
from numpy import random as nr

a = np.arange(1, 25, dtype = float)
c1 = nr.choice(a, size = (3, 4))
c2 = nr.choice(a, size = (3, 4), replace=False)
#下式中参数p指定每个元素对应的抽取概率，默认为每个元素被抽取的概率相同
c3 = nr.choice(a, size = (3, 4), p = a / np.sum(a))
print("随机可重复抽取")
print(c1)
print("随机但不重复抽取")
print(c2)
print("随机但按指定概率抽取")
print(c3)
```

```shell
随机可重复抽取
[[18.  1.  7. 14.]
 [22. 23. 24.  1.]
 [24.  9.  8. 10.]]
随机但不重复抽取
[[ 3. 20.  4. 10.]
 [13. 22.  5. 17.]
 [ 8. 15. 24.  2.]]
随机但按指定概率抽取
[[20.  5. 12. 18.]
 [18. 15.  7. 14.]
 [23. 21. 24. 18.]]
```

### 1.3 矩阵操作

深度学习中经常涉及多维数组或矩阵的运算，正好NumPy模块提供了许多相关的计算方法，下面介绍一些常用的方法。

```shell
import numpy as np

nd14 = np.arange(9).reshape([3,3])
print(nd14)

#矩阵转置
print(np.transpose(nd14))

#矩阵乘法
a = np.arange(12).reshape([3, 4])
print(a)
b = np.arange(8).reshape([4, 2])
print(b)
print(a.dot(b))

#求矩阵的迹
print(nd14.trace())
#计算矩阵行列式
print(np.linalg.det(nd14))

#计算逆矩阵
c = np.random.random([3, 3])
print(c)
print(np.linalg.solve(c, np.eye(3)))
```

```shell
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[0 3 6]
 [1 4 7]
 [2 5 8]]
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
[[ 28  34]
 [ 76  98]
 [124 162]]
12
0.0
[[0.49004436 0.00303589 0.84366215]
 [0.54368872 0.13869147 0.08572875]
 [0.40161209 0.82697863 0.80616256]]
[[ 0.11650638  1.97986434 -0.33246808]
 [-1.15011756  0.16012924  1.18658819]
 [ 1.12177409 -1.15058794  0.18884545]]
```

上面介绍的几种是numpy.linalg模块中的函数，numpy.linalg模块中的函数是满足行业标准级的Fortan库。

| 函数  |                说明                |
| :---: | :--------------------------------: |
| diag  | 以一维数组方式返回方针的对角线元素 |
|  dot  |              矩阵乘法              |
| trace |     求迹，即计算对角线元素的和     |
|  det  |            计算矩阵列式            |
|  eig  |     计算方阵的特征值和特征向量     |
|  inv  |            计算方阵的逆            |
|  qr   |             计算qr分解             |
|  svd  |         计算奇异值分解svd          |
| solve |   解线性方程式Ax=b，其中A为方阵    |
| lstsq |        计算Ax=b的最小二乘解        |

### 1.4 数据合并与展平

在机器学习或深度学习中，会经常遇到需要把多个向量或矩阵按某轴方向进行合并的情况，也会遇到需要展平的情况，如在卷积或循环神经网络中，在全连接层之前，需要把矩阵展平。

#### 1.合并一维数组

```shell
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)
print(c)
d = np.concatenate([a, b])
print(d)
```

```shell
[1 2 3 4 5 6]
[1 2 3 4 5 6]
```

#### 2.多维数组的合并

```shell
import numpy as np

a = np.arange(4).reshape([2, 2])
b = np.arange(4).reshape([2, 2])

#按行合并
c = np.append(a, b, axis=0)
print(c)
print("合并后数据维度", c.shape)

#按列合并
d = np.append(a, b, axis=1)
print(d)
print("合并后数据维度", d.shape)
```

```shell
[[0 1]
 [2 3]
 [0 1]
 [2 3]]
合并后数据维度 (4, 2)
[[0 1 0 1]
 [2 3 2 3]]
合并后数据维度 (2, 4)
```

#### 3.矩阵展平

```shell
import numpy as np

nd15 = np.arange(6).reshape(2, -1)
print(nd15)
#按照列优先，展平
print("按列优先,展平")
print(nd15.ravel('F'))
#按照行优先，展平
print("按行优先,展平")
print(nd15.ravel())
```

```she
[[0 1 2]
 [3 4 5]]
按列优先,展平
[0 3 1 4 2 5]
按行优先,展平
[0 1 2 3 4 5]
```

### 1.5 通用函数

NumPy提供了两种基本的对象，即ndarray和ufunc对象。ufunc（通用函数）是universal function的缩写，它是一种能对数组的每个元素进行操作的函数。许多ufunc函数都是在C语言级别实现的，因此它们的计算速度非常快。此外，功能比math模块中的函数更灵活。math模块的输入一般是标量，但NumPy中的函数可以是向量或矩阵，而利用向量或矩阵可以避免循环语句，这点在机器学习、深度学习中经常使用。

|       函数        |         使用方法         |
| :---------------: | :----------------------: |
|       sqrt        |  计算序列化数据的平方根  |
|      sin,cos      |         三角函数         |
|        abs        |  计算序列化数据的绝对值  |
|        dot        |         矩阵运算         |
|  log,log10,log2   |         对数函数         |
|        exp        |         指数函数         |
| cumsum,cumproduct |      累计求和，求积      |
|        sum        | 对一个序列化数据进行求和 |
|       mean        |         计算均值         |
|      median       |        计算中位数        |
|        std        |        计算标准差        |
|        var        |         计算方差         |
|     corrcoef      |       计算相关系数       |

#### 1.使用math与numpy函数性能比较

```shell
import time
import math
import numpy as np

x = [i * 0.001 for i in np.arange(1000000)]
start = time.clock()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print("math.sin:", time.clock() - start)

x = [i * 0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.clock()
np.sin(x)
print("numpy.sin:", time.clock() - start)
```

```she
math.sin: 0.4730549999985669
numpy.sin: 0.018625799999426818
```

#### 2.使用循环与向量运算比较

充分利用Python的NumPy库中的内建函数（built-in function），实现计算的向量化，可大大提高运行速度。NumPy库中内建函数使用了SIMD指令。在Python中使用向量化要比使用循环计算速度快得多。

```she
import numpy as np

x1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)

##使用循环计算向量点积
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()    
print("dot = " + str(dot) + "\n for loop----- Computation time = " + str(1000*(toc - tic)) + "ms")

##使用numpy函数求点积
tic = time.process_time()
dot = 0
dot = np.dot(x1, x2)
toc = time.process_time()    
print("dot = " + str(dot) + "\n verctor version----- Computation time = " + str(1000*(toc - tic)) + "ms")
```

```shell
dot = 249695.95496750838
 for loop----- Computation time = 812.5ms
dot = 249695.95496750379
 verctor version----- Computation time = 0.0ms
```

### 1.6 广播机制

广播机制（Broadcasting）的功能是为了方便不同shape的数组（Numpy库的核心数据结构）进行数学运算。广播提供了一种向量化数组操作的方法，以便在C中而不是在Python中进行循环，这通常会带来更高效的算法实现。广播的兼容原则为：

* 对齐尾部维度
* shape相等或其中shape元素中有一个为1

```shell
import numpy as np

a = np.arange(10)
b = np.arange(10)
#两个shape相同的数组相加
print(a + b)
#一个数组与标量叠加
print(a + 3)
#两个向量相乘
print(a * b)

#多维数组之间的运算
c = np.arange(10).reshape(5, 2)
d = np.arange(2).reshape(1, 2)
#将d数组进行复制扩充为【5,2】
print(c + d)
```

```shell
[ 0  2  4  6  8 10 12 14 16 18]
[ 3  4  5  6  7  8  9 10 11 12]
[ 0  1  4  9 16 25 36 49 64 81]
[[ 0  2]
 [ 2  4]
 [ 4  6]
 [ 6  8]
 [ 8 10]]
```

### 1.7 小结

## 第2章 Theano基础

NumPy是数据计算的基础，更是深度学习框架的基石。但如果直接使用NumPy计算大数据，其性能已成为一个瓶颈。

随着数据爆炸式增长，尤其是图像数据，音频数据等数据的快速增长，迫切需要突破NumPy性能上的瓶颈。

Theano是Python的一个库，为开源项目。对于解决大量数据的问题，使用Theano可能获得与手工用C实现差不多的性能。另外通过利用GPU，它能获得比CPU上快很多数量级的性能。

### 2.1 安装

#### 1.安装anaconda

安装完成后，可用conda list命令查看已安装的库：

```shell
conda list
```

#### 2.安装Theano

利用conda来安装或更新程序

```shell
conda install theano
```

#### 3.测试

先启动Python，然后导入theano模块，如果不报错，说明安装成功。

```shell
import theano
```

### 2.2 符号变量

存储数量需要用到各种变量，Theano用符号变量TensorVariable来表示变量，又称为张量（Tensor）。张量是Theano的核心元素（也是TensorFlow的核心元素），是Theano表达式和运算操作的基本单位。张量是标量（scalar）、向量（vector）、矩阵（matrix）等的统称。具体来说，标量就是我们通常看到的0阶的张量，如12，a等，而向量和矩阵分别为1阶张量和2阶的张量。

首先定义三个标量：一个代表输入x、一个代表权重w、一个代表偏移量b，然后计算这些标量结果z=x*w+b,Theano代码实现如下：

```shell
import theano
from theano import tensor as T

#初始化张量
x = T.scalar(name = 'input', dtype = 'float32')
w = T.scalar(name = 'weight', dtype = 'float32')
b = T.scalar(name = 'bias', dtype = 'float32')
z = w * x + b

#编译程序
net_input = theano.function(inputs = [w, x, b], outputs = z)
#执行程序
print('net_input: %2f'% net_input(2.0, 3.0, 0.5))
```

```she
net_input: 6.500000
```

Theano本身是一个通过的符号计算框架，与非符号架构的框架不同，它先使用tensor variable初始化变量，然后将复杂的符号表达式编译成函数模型，最后运行时传入实际数据进行计算。整个过程涉及三个步骤：定义符号变量，编译代码，执行代码。

#### 1.使用内置的变量类型创建

目前Theano支持7中内置的变量类型，分别是标量（scalar）、向量（vector）、行（row）、列（col）、矩阵（matrix）、tensor3、tensor4等。其中标量是0阶张量，向量为1阶张量，矩阵为2阶张量等。

```shell
import theano
from theano import tensor as T

x = T.scalar(name = 'input', dtype = 'float32')
data = T.vector(name = 'data', dtype = 'float64')
```

#### 2.自定义变量类型

内置的变量类型只能处理4维及以下的变量，如果需要处理更高维的数据时，可以使用Theano的自定义变量类型，具体通过TensorType方法来实现：

```shell
import theano
from theano import tensor as T

mytype = T.TensorType('float64', broadcastable = (), name = None, sparse_grad = False)
```

其中broadcastable是True或False的布尔类型元组，元组的大小等于变量的维度，如果为True，表示变量在对应维度上的数据可以进行广播，否则数据不能广播。

广播机制（broadcast）是一种重要机制，有了这种机制，就可以方便对不同维的张量进行运算，否则，就要手工把低维数据变成高维，利用广播机制系统自动复制等方法把低维数据补齐（NumPy也有这种机制）。

```shell
import theano
import numpy as np
import theano.tensor as T

r = T.row()
print(r.broadcastable)

mtr = T.matrix()
print(mtr.broadcastable)

f_row = theano.function([r, mtr], [r + mtr])
R = np.arange(1,3).reshape(1, 2)
print(R)

M = np.arange(1, 7).reshape(3, 2)
print(M)

f_row(R, M)
```

```shell
(True, False)
(False, False)
[[1 2]]
[[1 2]
 [3 4]
 [5 6]]
[array([[2., 4.],
        [4., 6.],
        [6., 8.]])]
```

#### 3.将Python类型变量或者NumPy类型变量转化为Theano共享变量

共享变量是Theano实现变量更新的重要机制。要创建一个共享变量，只要把一个Python对象或NumPy对象传递给shared函数即可。

```shell
import theano
import numpy as np
import theano.tensor as T

data = np.array([[1, 2], [3, 4]])
shared_data = theano.shared(data)
print(type(shared_data))
```

```shell
<class 'theano.tensor.sharedvar.TensorSharedVariable'>
```

### 2.3 符号计算图模型

符号变量定义后，需要说明这些变量间的运算关系。Theano实际采用符号计算图模型来实现。首先创建表达式所需的变量，然后通过操作符（op）把这些变量结合在一起。

Theano处理符号表达式时是通过把符号表达式转换为一个计算图（graph）来处理（TensorFlow也使用了这种方法），符号计算图的节点有：variable、type、apply和op。

* variable节点：即符号的变量节点，符号变量是符号表达式存放信息的数据结构，可以分为输入符号和输出符号。
* type节点：当定义了一种具体的变量类型以及变量的数据类型时，Theano为其指定数据存储的限制条件。
* apply节点：把某一个类型的符号操作符应用到具体的符号变量中，与variable不同，apply节点无需由用户指定，一个apply节点包括3个字段：op、inputs、outputs。
* op节点：即操作符节点，定义了一种符号变量间的运算，如+、—、sum()、tanh()等。

Theano是将符号表达式的计算表示成计算图。这些计算图是由Apply和Variable将节点连接而组成，它们分别与函数的应用和数据相连接。操作由op实例表示，而数据类型由type实例表示。

```shell
import theano
import numpy as np
import theano.tensor as T

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
```

![1595329746332](C:\Users\ADMINI~1\AppData\Local\Temp\1595329746332.png)

箭头表示指向Python对象的引用。中间大的长方形是一个Apply节点，3个圆角矩形（如X）是Variable节点，带+号的圈圈是ops，3个圆角小长方形（如matrix）是Types。

在创建Variables之后，应用Apply ops得到更多的变量，这些变量仅仅是一个占位符，在function中作为输入。变量指向Apply节点的过程是用来表示函数通过owner域来生成它们。这些Apply节点是通过它们的inputs和outputs域来得到它们的输入和输出变量。

x和y的owner域的指向都是None，这是因为它们不是另一个计算的结果。如果它们中的一个变量是另一个计算的结果，那么owner域将会指向另一个蓝色盒。

### 2.4 函数

函数是Theano的一个核心设计模块，它提供一个接口，把函数计算图编译为可调用的函数对象。

#### 1.函数定义的格式

```shell
#函数格式示例
theano.function(inputs, outputs, mode=None, updates=None, givens=None, no_default_updates=False, accept_inplace=False, name=None, rebuild_strict=True, allow_input_downcast=None, profile=None, on_unused_input='raise')
```

一般只用到三个参数：inputs表示自变量；outputs表示函数的因变量（也就是函数的返回值）；还有一个比较常用的是updates参数，它一般用于神经网络共享变量参数更新，通常以字典或元组列表的形式指定。此外，givens是一个字典或元组列表，记为[(var1,var2)]，表示在每一次函数调用时，在符号计算图中，把符号变量var1节点替换为var2节点，该参数常用来指定训练数据集的batch大小。

```shell
import theano

x, y = theano.tensor.fscalars('x', 'y')
z1 = x + y
z2 = x * y
#定义x、y为自变量，z1、z2为函数返回值（因变量）
f = theano.function([x, y], [z1, z2])

#返回当x=2，y=3的时候，函数f的因变量z1，z2的值
print(f(2, 3))
```

```shell
[array(5., dtype=float32), array(6., dtype=float32)]
```

在执行theano.function()时，Theano进行了编译优化，得到一个end-to-end的函数，传入数据通过调用f(2,3)时，执行的是优化后保存在图结构中的模型，而不是我们写的那行z=x+y，尽管二者结果一样。这样的好处是Theano可以对函数f进行优化，提升速度；坏处是不方便开发和调试，由于实际执行的代码不是我们写的代码，所以无法设置断点进行调试，也无法直接观察执行时中间变量的值。

#### 2.自动求导

有了符号计算，自动计算导数就很容易了。tensor.grad()唯一需要做的就是从outputs逆向遍历到输入节点。对于每个op，它都定义了怎么根据输入计算出偏导数。使用链式法则就可以计算出梯度了。利用Theano求导时非常方便，可以直接利用函数theano.grad()。比如求s函数的导数：

*s*(x) = 1 / (1 + exp(-x))

以下代码实现当x=3的时候，求s函数的导数：

```shell
import theano

x = theano.tensor.fscalar('x') #定义一个float类型的变量x
y = 1 / (1 + theano.tensor.exp(-x)) #定义变量y
dx = theano.grad(y, x) #偏导数函数
f = theano.function([x], dx) #定义函数f，输入为x，输出为s函数的偏导数
print(f(3)) #计算当x=3的时候，函数y的偏导数
```

```shell
0.04517666
```

#### 3.更新共享变量参数

在深度学习中通常需要迭代多次，每次迭代都需要更新参数。在theano.function函数中，有一个非常重要的参数updates。updates是一个包含两个元素的列表或元组，一般示例为updates=[old_w,new_w]，当函数被调用的时候，会用new_w替换old_w。

```shell
import theano

w = theano.shared(1) #定义一个共享变量w，其初始化值为1
x = theano.tensor.iscalar('x')
f = theano.function([x], w, updates=[[w, w+x]]) #定义函数自变量为x，因变量为w，当函数执行完毕后，更新参数w=w+x
print(f(3)) #函数输出为w
print(w.get_value()) #这个时候可以看到w=w+x为4
```

```shell
1
4
```

在求梯度下降的时候，经常用到updates这个参数。比如updates=[w, w-α*(dT/dw)]，其中dT/dw就是梯度下降时，代价函数对参数w的偏导数，α是学习速率。下举一逻辑回归的完整实例来说明：

```shell
import numpy as np
import theano
import theano.tensor as T

rng = np.random

#我们为了测试，自己生成10个样本，每个样本是3维的向量，然后用于训练
N = 10
feats = 3
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N, low=0, high=2).astype(np.float32))

#声明自变量x、以及每个样本对应的标签y（训练标签）
x = T.matrix("x")
y = T.vector("y")

#随机初始化参数w、b=0，为共享变量
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

#构造代价函数
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b)) #s激活函数
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) #交叉熵代价函数
cost = xent.mean() + 0.01 * (w ** 2).sum() #代价函数的平均值+L2正则项以防过拟合，其中权重衰减系数为0.01
gw, gb = T.grad(cost, [w, b]) #对总代价函数求参数的偏导数

prediction = p_1 > 0.5 #大于0.5预测值为1，否则为0
train = theano.function(inputs=[x, y], outputs=[prediction, xent], updates=((w, w-0.1*gw), (b, b-0.1*gb))) #训练所需函数
predict = theano.function(inputs=[x], outputs=prediction) #测试阶段函数

#训练
training_steps = 1000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print(err.mean()) #查看代价函数下降变化过程
```

### 2.5 条件与循环

#### 1.条件判断

Theano是一种符号语言，条件判断不能直接使用Python的if语句。在Theano可以用ifelse和switch来表示判定语句。

switch对每个输出变量进行操作，ifelse只对一个满足条件的变量操作。比如

对语句：

```shell
switch(cond, ift, iff)
```

如果满足条件，则switch既执行ift也执行iff。而对语句：

```shell
if cond then ift else iff
```

ifelse只执行ift或者只执行iff。

```shell
from theano import tensor as T
from theano.ifelse import ifelse
import theano,time,numpy

a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')
z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y)) #lt:a<b?
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

#optimizer:optimizer的类型结构
#linker:决定使用哪种方式进行编译
f_switch = theano.function([a, b, x, y], z_switch, mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a, b, x, y], z_lazy, mode=theano.Mode(linker='vm'))

val1 = 0.
val2 = 1.

big_mat1 = numpy.ones((1000, 100))
big_mat2 = numpy.ones((1000, 100))

n_times = 10

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print('time spent evaluating both values %f sec' % (time.clock() - tic))

tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))
```

```shell
time spent evaluating both values 0.003216 sec
time spent evaluating one value 0.004673 sec
```

#### 2.循环语句

scan是Theano中构建循环Graph的方法，scan是个灵活复杂的函数，任何用循环、递归或者跟序列有关的计算，都可以用scan完成。

```shell
theano.scan(fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False, mode=None, name=None, profile=False, allow_gc=None, strict=False)
```

参数说明：

* fn：一个lambda或者def函数，描述了scan中的一个步骤。除了outputs_info，fn可以返回sequences变量的更新updates。fn的输入变量的顺序为sequences中的变量、outputs_info的变量、non_sequences中的变量。如果使用了taps，则按照taps给fn喂变量。
* sequences：scan进行迭代的变量，scan会在T.arange()生成的list上遍历。
* outputs_info：初始化fn的输出变量，和输出的shape一致。如果初始化值设为None，表示这个变量不需要初始值。
* non_sequences：fn函数用到的其他变量，迭代过程中不可以改变（unchange）。
* n_steps：fn的迭代次数。

先定义函数one_step，即scan里的fn，其任务就是计算多项式的一项，scan函数返回的result里会保存多项式每一项的值，然后我们对result求和，就得到了多项式的值。

```shell
import theano
import theano.tensor as T
import numpy as np

#定义单步的函数，实现a*x^n
#输入参数的顺序要与下面scan的输入参数对应
def one_step(coef, power, x):
    return coef * x ** power

coefs = T.ivector() #每步变化的值，系数组成的向量
powers = T.ivector() #每步变化的值，指数组成的向量
x = T.iscalar() #每步不变的值，自变量

#seq，out_info，non_seq与one_step函数的参数顺序一一对应
#返回的result是每一项的符号表达式组成的list
result, updates = theano.scan(fn = one_step, sequences = [coefs, powers], outputs_info = None, non_sequences = x)

#每一项的值与输入的函数关系
f_poly = theano.function([x, coefs, powers], result, allow_input_downcast = True)

coef_val = np.array([2, 3, 4, 6, 5])
power_val = np.array([0, 1, 2, 3, 4])
x_val = 10

print("多项式各项的值：", f_poly(x_val, coef_val, power_val))
#scan返回的result是每一项的值，并没有求和，如果我们只想要多项式的值，可以把f_poly写成这样：
#多项式每一项的和与输入的函数关系
f_poly = theano.function([x, coefs, powers], result.sum(), allow_input_downcast = True)

print("多项式和的值：", f_poly(x_val, coef_val, power_val))
```

```shell
多项式各项的值： [    2    30   400  6000 50000]
多项式和的值： 56432
```

### 2.6 共享变量

共享变量（shared variable）是实现机器学习算法参数更新的重要机制。shared函数会返回共享变量。这种变量的值在多个函数可直接共享。可以用符号变量的地方都可以用共享变量。但不同的是，共享变量有一个内部状态的值，这个值可以被多个函数共享。它可以存储在显存中，利用GPU提高性能。我们可以使用get_value和set_value方法来读取或者修改共享变量的值，使用共享变量实现累加操作。

```shell
import theano
import theano.tensor as T
from theano import shared
import numpy as np

#定义一个共享变量，并初始化为0
state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state + inc)])
#打印state的初始值
print(state.get_value())
accumulator(1) #进行一次函数调用
#函数返回后，state的值发生了变化
print(state.get_value())
```

这里state是一个共享变量，初始化为0，每次调用accumulator()，state都会加上inc。共享变量可以像普通张量一样用于符号表达式，另外，它还有自己的值，可以直接用.get_value()和.set_value()方法来访问和修改。

上述代码引入了函数中的updates参数。updates参数是一个list，其中每个元素是一个元组（tuple），这个tuple的第一个元素是一个共享变量，第二个元素是一个新的表达式。updates中的共享变量会在函数返回后更新自己的值。updates的作用在于执行效率，updates多数时候可以用原地（in-place）算法快速实现，在GPU上，Theano可以更好地控制何时何地给共享分配空间，带来性能提升。

### 2.7 小结

Theano基于NumPy，但性能方面又高于NumPy。因Theano采用了张量（Tensor）这个核心元素，在计算方面采用符号计算模型，而且采用共享变量、自动求导、利用GPU等适合于大数据、深度学习的方法，其他很多开发项目也深受这些技术和框架影响。

## 第3章 线性代数

机器学习、深度学习的基础除了编程语言外，还有一个就是应用数学。它一般包括线性代数、概率与信息论、概率图、数值计算与最优化等。其中线性代数有事基础的基础。线性代数是数学的一个重要分支，广泛应用于科学和工程领域。大数据、人工智能的源数据在模型训练前都需要转换为向量或矩阵，而这些运算正是线性代数的主要内容。

### 3.1 标量、向量、矩阵和张量

在机器学习、深度学习中，首先遇到的就是数据，如果按类别来划分，通常会遇到以下几种类型的数据。

#### 1.标量（scalar）

一个标量就是一个单独的数，一般用小写的变量名称来表示，如a，x等

#### 2.向量（vector）

向量就是一列数或一个一维数组，这些数都是有序排列的。通过次序中的索引，我们可以确定向量中每个单独的数。通常我们赋予向量粗体的小写变量名称，如***x***,***y***等。一个向量一般有很多元素，我们一般通过带脚标的斜体表示，如*x*1表示向量***x***中的第一个元素，*x*2表示第二个元素，依次类推。

当需要明确表示向量中的元素时，我们一般将元素排列成一个方括号包围的纵列：

***x*** = [*x*1,*x*2...,*x*n]

我们可以把向量看做空间中的点，每个元素是不同的坐标轴上的坐标。

```shell
import numpy as np

a = np.array([1, 2, 4, 3, 8])
print(a.size)
print(a[0], a[1], a[2], a[-1])
```

```shell
5
1 2 4 8
```

这说明向量元素个数为5，向量中索引一般从0开始，如a[0]表示第一个元素1，a[1]表示第二个元素2，a[2]表示第三个元素4，依次类推。这是从左到右的排列顺序，如果从右到左，我们可用负数来表示，如a[-1]表示第1个元素（从右往左），a[-2]表示第2个元素，依次类推。

#### 3.矩阵（matrix）

矩阵是二维数组，其中的每一个元素被两个索引而非一个所确定。我们通常会比如赋予矩阵粗体的大写变量名称，比如***A***。如果一个实数矩阵高度为*m*，宽度为*n*，那么我们说***A***∈Rm*n。

与向量类似，可以通过给定行和列的下标表示矩阵中的单个元素，下标用逗号分隔，如***A***1,1表示***A***左上的元素，***A***1,2表示第一行第二列对应的元素，依次类推。如果我们想表示1列或1行，我们可以引入冒号":"来表示，如第一行，可用***A***1,:表示，第2行用***A***2,:表示，第一列用***A***:,1表示，第n列用***A***:,n表示。

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print(A.size)
print(A.shape)
print(A[0,0], A[0,1], A[1,1])
print(A[1,:])
print(A[:,1])
```

```shell
[[1 2 3]
 [4 5 6]]
6
(2, 3)
1 2 5
[4 5 6]
[2 5]
```

矩阵可以用嵌套向量生成，和向量一样，在NumPy中，矩阵元素的下标索引也是从0开始的。

#### 4.张量（tensor）

几何代数中定义的张量是向量和矩阵的推广，通俗一点理解的话，我们可以将标量视为零阶张量，向量视为一阶张量，那么矩阵就是二阶张量，三阶就称为三阶张量，以此类推。在机器学习、深度学习中经常遇到多维矩阵，如一张彩色图片就是一个三阶张量，三个维度分别是图片的高度、宽度和色彩数据。

张量（tensor）也是深度学习框架TensorFlow的重要概念。TensorFlow由tensor（张量）+ flow（流）构成。

```python
import numpy as np

B = np.arange(16).reshape((2, 2, 4))
print(B)
print(B.size)
print(B.shape)
print(B[0,0,0], B[0,0,1], B[0,1,1])
print(B[0,1,:])
```

```python
[[[ 0  1  2  3]
  [ 4  5  6  7]]

 [[ 8  9 10 11]
  [12 13 14 15]]]
16
(2, 2, 4)
0 1 5
[4 5 6 7]
```

#### 5.转置（transpose）

转置以主对角线（左上到右下）为轴进行镜像操作，通俗一点来说就是行列互换。将矩阵***A***转置表示为***A***T，定义如下：

(***A***T)i,j = ***A***j,i

```python
import numpy as np

C = np.array([[1,2,3], [4,5,6]])
D = C.T
print(C)
print(D)
```

```python
[[1 2 3]
 [4 5 6]]
[[1 4]
 [2 5]
 [3 6]]
```

### 3.2 矩阵和向量计算

矩阵加法和乘法是矩阵运算中最常用的操作之一，两个矩阵相加，需要它们的形状相同，进行对应元素的相加，如***C=A+B***，其中Ci,j = Ai,j + Bi,j。矩阵也可以和向量相加，只要它们的列数相同，相加的结果是矩阵每行与向量相加，这种隐式复制向量b到很多位置的方法称为广播（broadcasting）。

```python
import numpy as np

C = np.array([[1,2,3], [4,5,6]])
b = np.array([10,20,30])
D = C + b
print(D)
```

```python
[[11 22 33]
 [14 25 36]]
```

两个矩阵相加，要求它们的形状相同，如果两个矩阵相乘，如***A***和***B***相乘，结果为矩阵***C***，那么只要矩阵***A***的列数和矩阵***B***的行数相同即可。如果矩阵***A***的形状为m * n，矩阵***B***的形状为n * p，那么矩阵***C***的形状就是m * p。

### 3.3 特殊矩阵与向量

#### 1.可逆矩阵

#### 2.对角矩阵

#### 3.对称矩阵

#### 4.单位向量

#### 5.正交向量

#### 6.正交矩阵

### 3.4 线性相关性及向量空间

#### 1.线性组合

#### 2.线性相关

#### 3.向量组的秩

### 3.5 范数

### 3.6 特征值分解

### 3.7 奇异值分解

### 3.8 迹运算

### 3.9 实例：用Python实现主成分分析

主成分分析（Principal Component Analysis，PCA）是一种统计方法。通过正交变换将一组可能存在相关性的变量转换为一组线性不相关的变量，转换后的这组变量叫主成分。