import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math

xArr=[]
yArr=[]
begin=1
while begin < 9:
    y = (16 - (begin-5)**2)**0.5
    y1 = 5 + y
    y2 = 5 - y
    xArr.append(begin)
    yArr.append(y1)
    xArr.insert(0, begin)
    yArr.insert(0, y2)
    begin+=0.1

xArr_=[]
yArr_=[]
begin=2
while begin < 8:
    y_ = (9 - (begin-5)**2)**0.5
    y1_ = 5 + y_
    y2_ = 5 - y_
    xArr_.append(begin)
    yArr_.append(y1_)
    xArr_.insert(0, begin)
    yArr_.insert(0, y2_)
    begin+=0.1

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xArr,yArr,[0 for i in range(len(xArr))], color="red",linewidth=2)
ax.plot(xArr_,yArr_,[0 for i in range(len(xArr_))], color="blue",linewidth=2)
#plt.plot(xArr,yArr,label="f(x)=-1 or 1",color="red",linewidth=2)
#plt.plot(xArr_,yArr_,label="f(x)=-1 or 1",color="blue",linewidth=2)

x=[]
y=[]
z=[]
sigma=0.05
for i in range(len(xArr)):
    com = math.exp(-sigma * xArr[i]**2) * math.exp(-sigma * yArr[i]**2)
    x.append(com)
    y.append(com * 2 * sigma * xArr[i] * yArr[i])
    z.append(com * 2 * sigma**2 * xArr[i]**2 * yArr[i]**2)

ax.plot(x,y,z, color="red",linewidth=1)

x=[]
y=[]
z=[]
for i in range(len(xArr_)):
    com = math.exp(-sigma * xArr_[i]**2) * math.exp(-sigma * yArr_[i]**2)
    x.append(com)
    y.append(com * 2 * sigma * xArr_[i] * yArr_[i])
    z.append(com * 2 * sigma**2 * xArr_[i]**2 * yArr_[i]**2)

ax.plot(x,y,z, color="blue",linewidth=1)

plt.show()
