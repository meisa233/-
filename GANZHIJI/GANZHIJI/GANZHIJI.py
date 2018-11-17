import numpy as np
from numpy.ma.core import inner
import matplotlib.pyplot as plt
x=np.array([(3,3),(4,3),(1,1)])#初始化训练样本
y=np.array([(1),(1),(-1)])#样本的输出正实例点和负实例点
#print (data_)
global w,b,rate
w=np.array([0,0]);b=0#初始化参数
rate=1#学习速率


def gz(x):#测试用
    if x>=0:
        return 1
    else:
        return -1

def update(x,y):#用于更新参数
        global w,b,rate# 必须有这句，否则无法在该函数中操作w,b和rate
        temp0=w+rate*y*x #用于更新w和b
        temp1=b+rate*y
        w=temp0
        b=temp1
        print("w:",w,"b",b)


i=0
flag=True
while flag==True:
    while i<=2:
        if y[i]*(inner(x[i],w)+b)<=0:#当出现误分类点时；inner用于计算向量内积
           print("误分类点：","x",i+1,"",end="")#最后的end=""用于输出时不用输出换行符（默认是会输出换行符的）
           update(x[i],y[i])
           i=0
           break
        i=i+1
    if i>2:
        flag=False

print("感知机求解结果：w=",w,";b=",b)

parameter=w.T
parameter_inv=np.linalg.pinv(parameter)

xandy=-b*parameter_inv

'''
x=[2,3]  
y=[0,1]  
#创建绘图对象  
plt.figure()  
#在当前绘图对象进行绘图（两个参数是x,y轴的数据）  
plt.plot(x,y)  
#保存图象  
#plt.savefig("easyplot.png") 
plt.show() 
'''