# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:14:08 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP

#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

plt.figure(figsize=(30,10), dpi=80)

x_3 = range(1, 32)
y_3 = [11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22,22,22,23]
x_10 = range(41, 72)
y_10 = [26,26,28,19,21,17,16,19,18,20,20,19,22,23,17,20,21,20,22,15,11,15,5,13,17,10,11,13,12,13,6]

#绘制散点图
plt.scatter(x_3, y_3, label="3月份")
plt.scatter(x_10, y_10, label="10月份")

#调整x轴的刻度
_x = list(x_3) + list(x_10)
_xticks_labels = ["3月{}日".format(i) for i in x_3]
_xticks_labels += ["10月{}日".format(i-40) for i in x_10]
plt.xticks(_x[::3], _xticks_labels[::3], fontproperties=font, rotation=45)

plt.xlabel("日期", fontproperties=font)
plt.ylabel("温度", fontproperties=font)
plt.title("3月、10月的温度散点图", fontproperties=font)

plt.grid()
plt.legend(prop=font)
#保存图片
plt.savefig("./散点图.png")

#展示图形
plt.show()