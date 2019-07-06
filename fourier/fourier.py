# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math
import sys
#约定俗成的写法plt
#首先定义两个函数（正弦&余弦）
import numpy as np

x=np.linspace(-10,10,1000)#X轴数据
y=[]
for i in range(len(x)):
    re = 4 / math.pi
    temp = 0
    # replace 10 to 50 or 1000...
    for j in  range(10):
        k = j + 1
        temp += np.sin((2 * k - 1) * x[i]) / (2 * k - 1)
    y.append(re * temp)
        
        
plt.figure(figsize=(8,4))

plt.plot(x,y,label="f(x)=-1 or 1",color="red",linewidth=2)#将$包围的内容渲染为数学公式

plt.ylim(-1.5,1.5)
plt.legend()#显示左下角的图例

plt.show()
