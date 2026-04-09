import os
import sys
import numpy as np
import time
#获得随机数来生成结构，用于FDTD计算
def read_txt(filename,mark,typ):
    file = open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    for i in range(len(a)):
        a[i]=a[i].split(mark)
    a=np.array(a)
    a=a.astype(typ)
    return(a)
def find(si,team):
    x=0
    y=0
    for i in team:
        for j in range(len(si)):
            x+=abs(i[j]-si[j])
        if x==0:
            y=1
            break
        x=0
    return(y)
#1 same 0 different
if __name__ == '__main__':
    #mater="Au (Gold) - Palik";
    sys.path.append(os.path.dirname(__file__))
    tolnum=10000
    singnum=20
    st=[]
    id=[i+1 for i in range(100)]
    for i in range(tolnum*2):
        np.random.shuffle(id)
        t=id[:20]
        t.sort()
        if i!=0:
            if find(t,st)==0:
                st.append(t)
        if len(st)==tolnum:
            break
    with open("D:\\PengPu\\QR_code\\st_test.txt", "a") as f:
        for j in range(tolnum):
            for i in range(singnum-1):
                f.write(str(st[j][i])+',')
            f.write(str(st[j][singnum-1])+'\n')
        f.close()
