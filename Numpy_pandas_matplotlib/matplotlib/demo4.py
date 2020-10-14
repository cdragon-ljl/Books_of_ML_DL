# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:43:52 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP

#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

a = ["战狼2","速度与激情8","功夫瑜伽","西游降魔篇","变形金刚5\n：最后的骑士","摔跤吧！爸爸","加勒比海盗5\n：死无对证","金刚\n：骷髅岛","极限特工\n：终极回归","生化危机6\n：终章","乘风破浪","神偷奶爸3","智取威虎山","大闹天竺","金刚狼3\n：殊死一战","蜘蛛侠\n：英雄归来","悟空传","银河护卫队2","情圣","新木乃伊"]
b = [56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,10.3,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]

#设置图片大小
plt.figure(figsize=(20,10), dpi=80)

#绘制柱状图
#纵向
plt.bar(range(len(a)), b, width=0.4)

#设置字符串到x轴
plt.xticks(range(len(a)), a, fontproperties=font, rotation=90)

#横向
plt.barh(range(len(a)), b,height=0.4, color='black')

#设置字符串到y轴
plt.yticks(range(len(a)), a, fontproperties=font)

#保存图片
plt.savefig("./柱状图.png")

#显示图片
plt.show()