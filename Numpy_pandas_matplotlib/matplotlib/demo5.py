# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:22:22 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP

#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

a = ["猩球崛起3：终极之战","敦刻尔克","蜘蛛侠：英雄归来","战狼2"]
b_14 = [2358,399,2358,362]
b_15 = [12357,156,2045,168]
b_16 = [15746,312,4497,319]

#设置图片大小
plt.figure(figsize=(20,10), dpi=80)

bar_width = 0.2

x_14 = list(range((len(a))))
x_15 = [i+bar_width for i in x_14]
x_16 = [i+bar_width*2 for i in x_14]
        
#绘制柱状图
#纵向
plt.bar(x_14, b_14, width=bar_width, color='r', label="9月14日")
plt.bar(x_15, b_15, width=bar_width, color='g', label="9月15日")
plt.bar(x_16, b_16, width=bar_width, color='b', label="9月16日")

#设置字符串到x轴
plt.xticks(x_15, a, fontproperties=font)

#设置图例
plt.legend(prop=font)

#保存图片
plt.savefig("./柱状图2.png")

#显示图片
plt.show()