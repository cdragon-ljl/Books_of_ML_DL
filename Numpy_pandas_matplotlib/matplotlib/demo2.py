# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:16:43 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP

#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

x = range(11, 31)
y1 = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
y2 = [1,0,3,1,2,2,3,3,2,1,2,1,1,1,1,1,1,1,1,1]

#设置图片大小
plt.figure(figsize=(20,10), dpi=80)

plt.plot(x, y1, label="自己", linestyle='-', color='black')
plt.plot(x, y2, label="同桌", color='r')

#设置x轴刻度
_xtick_labels = ["{}岁".format(i) for i in x]
plt.xticks(x, _xtick_labels, fontproperties=font)

#绘制网格
plt.grid()

#添加图例
plt.legend(prop=font, loc=2)

plt.show()