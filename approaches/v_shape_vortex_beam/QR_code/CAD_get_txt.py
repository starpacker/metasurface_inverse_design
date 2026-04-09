import numpy as np
import math
from pyautocad import Autocad, APoint
#根据相位阵列的需求获得结构列表
def read_data(filename,num,type):
    file=open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    c=[]
    for x in a:
        b=x.split(',')
        c.append(b[:num])
    c=np.array(c)
    c=c.astype(type)
    return(c)
def edge(x,y,id):
    l=len(id)
    if x==l-1:
        right=0
    else:
        right=id[y][x+1]
    right=abs(id[y][x]-right)
    if x==0:
        left=0
    else:
        left=id[y][x-1]
    left=abs(id[y][x]-left)
    if y==0:
        up=0
    else:
        up=id[y-1][x]
    up=abs(id[y][x]-up)
    if y==l-1:
        down=0
    else:
        down=id[y+1][x]
    down=abs(id[y][x]-down)
    return(up,down,left,right)
if __name__ == '__main__':
    acad=Autocad(create_if_not_exists=True)
    l=125
    id=[[0 for i in range(l)]for j in range(l)]
    for y in range(l):
        for x in range(l):
            if (y-l/2+0.5)**2+(x-l/2+0.5)**2<=(l**2)/4:
                id[y][x]=1
    Xre,Y,Xpi,Xid=[],[],[],[]
    St=read_data('1064_dataset.txt',20,'int')
    Id=read_data('1064_id_dataset.txt',1,'float32')
    with open('st_direction_25.txt','a') as f:
        f.truncate(0)
        for y in range(l):
            for x in range(l):
                if id[y][x]!=0:
                    angle=math.atan2((x-l/2+0.5),(l/2-y-0.5))
                    angle=int((angle/np.pi+1)*180)
                    if angle==360:
                        angle=0
                    st=St[int(Id[angle])]
                    up0,down0,left0,right0=edge(x,y,id)
                    f.write(str(x)+','+str(y)+','+str(up0)+','+str(down0)+','+str(left0)+','+str(right0)+',')
                    for i in range(19):
                        f.write(str(st[i])+',')
                    f.write(str(st[19])+'\n')
        f.close()
