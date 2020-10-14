# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:42:48 2020

@author: cdragon-ljl
"""


from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP

#添加中文字体
font = FP(fname=r"c:\windows\fonts\simsun.ttc", size=15)

#设置图片大小
plt.figure(figsize=(15,8), dpi=80)

interval = [0,5,10,15,20,25,30,35,40,45,60,90,150]
width = [5,5,5,5,5,5,5,5,5,15,30,60]
quantity = [836,2737,3723,3926,3596,1438,3273,642,824,613,215,47]

plt.bar(range(len(quantity)), quantity, width=1)

_x = [i-0.5 for i in range(13)]
_xticks_labels = interval
plt.xticks(_x, _xticks_labels)

plt.grid()

plt.show()