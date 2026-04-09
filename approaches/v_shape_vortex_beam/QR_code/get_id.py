#根据多波长的FDTD数据，提取特定波长的响应，利用jones matrix计算出射光，并归一化，获得归一化后的单波长响应txt
#opr:Exx, Eyy, Exy, Eyx, phase
#st:l1,l2,a,b
import numpy as np
import matplotlib.pyplot as plt
import math
def read_data(filename):
    file=open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    c=[]
    for x in a:
        b=x.split(',')
        c.append(b)
    c=np.array(c)
    c=c.astype('float32')
    return(c)
def fold(x):
    for i in range(10):
        if x>np.pi:
            x=x-2*np.pi
        if x<-np.pi:
            x=x+2*np.pi
        if abs(x)<=np.pi:
            break
    return(x)
if __name__ == '__main__':
    pinchdata='D:\\PengPu\\QR_code\\'
    st,opr=[],[]
    wavelength=1064
    datanum=10000
    T=read_data(pinchdata+'1064_dataset.txt')
    Plr,Amlr=[],[]
    sum = 0
    for i in range(datanum):
        Plr.append(int((T[i][20]/np.pi+1)*180))
        Amlr.append(T[i][22])
        sum+=T[i][22]
    ave=sum/datanum
    for i in range(datanum):
        Amlr[i]-=ave

    idd=[]
    for i in range(360):
        amid=100
        for j in range(10000):
                if amid>abs(Amlr[j]):
                    amid=abs(Amlr[j])
                    idj=j
        idd.append(idj)

    with open (pinchdata+str(wavelength)+'_id_dataset.txt','a') as f:
        f.truncate(0)
        for i in range(len(idd)):
            f.write(str(idd[i])+'\n')
    print(len(idd))
