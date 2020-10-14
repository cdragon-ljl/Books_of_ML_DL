# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:45:54 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import numpy as np


#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

#设置图片大小
plt.figure(figsize=(15,8), dpi=80)

a = np.random.randint(75, high=148, size=250)
print(a)

#计算组数
d = 3 #组距
num_bins = (max(a) - min(a)) // d
print(max(a), min(a), max(a) - min(a))
print(num_bins)

#绘制直方图
plt.hist(a, num_bins)

#设置x轴刻度
plt.xticks(range(min(a), max(a)+d, d))

#设置网格
plt.grid(alpha=0.4)

#保存图片
plt.savefig("./直方图.png")

#绘制图形
plt.show()