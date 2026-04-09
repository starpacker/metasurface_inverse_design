import numpy as np
import math
from pyautocad import Autocad, APoint
#读取txt并画图，去除十字交叉
#unit cell边缘预留沟道
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
def get_location(T,x,y):
    j=-3
    for i in range(len(T)):
        if x==T[i][0] and y==T[i][1]:
            j=i
    return(j)
def single(T,id):
    sida = T[id]
    x = sida[0]
    y = sida[1]
    stru='0100100111101111011111000000111101010110000001000110111001000111101111011010111010111101111110111100'
    # st=st.split('')
    #st = sida[2:]

    st = []
    for i in range(len(stru)):
        substring = stru[i:i + 1]
        st.append(substring)


    # xc = (x - 62) * 0.41
    # yc = -(y - 62) * 0.41

    xc=0
    yc=0

    # Pix 1为无结构，Lx,1为有线
    Pix = [[0 for i in range(10)] for j in range(10)]
    Lx = [[0 for i in range(12)] for j in range(13)]
    Ly = [[0 for i in range(12)] for j in range(13)]

    for i in range(100):
        if st[i]==1:
            y0=i//10
            x0=i%10
            Pix[y0][x0]=1
    for i in range(10):
        for j in range(10):
            if Pix[i][j]==1:
                Lx[i+1][j+1]=1-Lx[i+1][j+1]
                Lx[i+2][j+1]=1-Lx[i+2][j+1]
                Ly[j+1][i+1]=1-Ly[j+1][i+1]
                Ly[j+2][i+1]=1-Ly[j+2][i+1]
    # 计算完毕开始画线
    for i in range(1, 12):
        for j in range(1, 11):
            if Lx[i][j] != 0:
                xl = (j - 6) * 0.04 + xc
                xr = xl + 0.04
                y1 = -(i - 6) * 0.04 + yc
                if Lx[i][j - 1] + Ly[j][i - 1] + Ly[j][i] == 3:
                    LU = APoint(xl, y1 + 0.002)
                    xl += 0.002
                    RD = APoint(xl, y1)
                    V = acad.model.AddLine(LU, RD)
                if Lx[i][j + 1] + Ly[j + 1][i - 1] + Ly[j + 1][i] == 3:
                    RD = APoint(xr, y1 - 0.002)
                    xr -= 0.002
                    LU = APoint(xr, y1)
                    V = acad.model.AddLine(LU, RD)
                L = APoint(xl, y1)
                R = APoint(xr, y1)
                V = acad.model.AddLine(L, R)
            if Ly[i][j] != 0:
                x2 = (i - 6) * 0.04 + xc
                yu = -(j - 6) * 0.04 + yc
                yd = yu - 0.04
                if Ly[i][j - 1] + Lx[j][i - 1] + Lx[j][i] == 3:
                    yu -= 0.002
                if Ly[i][j + 1] + Lx[j + 1][i - 1] + Lx[j + 1][i] == 3:
                    yd += 0.002
                U = APoint(x2, yu)
                D = APoint(x2, yd)
                V = acad.model.AddLine(U, D)
def conti(T,stri):
    stri=stri.replace(' ','')
    stri=stri.split('Y=')
    xar=float(stri[0])
    yar=float(stri[1])
    xar=int(xar/0.4+l/2)
    yar=int(l/2-yar/0.4)
    return(get_location(T,xar,yar))
if __name__ == '__main__':
    acad=Autocad(create_if_not_exists=True)
    T=read_data('st_direction_25.txt',26,'int')
    l=125
    ali=conti(T,'15.7600  Y=  -0.3200 ')

    for i in range(len(T)-ali):
        i=i+ali
        single(T,i)
