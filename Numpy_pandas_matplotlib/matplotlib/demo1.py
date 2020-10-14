# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:14:12 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


#添加中文字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

#设置图片大小
#fig = plt.figure(figsize=(20, 8), dpi=80)

x = range(2, 26, 2)
y = [15,13,14.5,17,28,25,26,26,24,22,18,15]

#绘图
plt.plot(x, y)

#设置x轴的刻度
plt.xticks(x)

#添加描述信息
plt.xlabel("时间", fontproperties=font)
plt.ylabel("温度", fontproperties=font)
plt.title("气温变化图", fontproperties=font)

#保存图片
plt.savefig("./1.png")

#展示图片
plt.show()