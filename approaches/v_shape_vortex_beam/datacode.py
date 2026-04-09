#根据多波长的FDTD数据，提取特定波长的响应，利用jones matrix计算出射光，并归一化，获得归一化后的单波长响应txt
#opr:Exx, Eyy, Exy, Eyx, phase
#st:l1,l2,a,b
import numpy as np
import matplotlib.pyplot as plt
import math
def read_data(filename,f):
    file = open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    st=a[len(a)-1]
    #l1 l2 a b
    st=st.split(',')
    st=np.array(st)
    st=st.astype('float32')
    f=int((f-530)*0.1)
    opr=a[f]
    #xx yy xy yx phase
    opr=opr.replace("i", "j", 5)
    opr = opr.split(',')
    opr=np.array(opr)
    opr=opr.astype('complex')
    return(st,opr)
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
    datapath='D:\\PengPu\\V\\multi_wavelength\\reflection\\mutiwavelength_dataset\\'
    pathout='D:\\PengPu\\V\\dataset_num\\'
    st,opr=[],[]
    wavelength=1064
    datanum=10000
    for i in range(datanum):
        fn=datapath+str(i)+'.txt'
        a,b=read_data(fn,wavelength)
        st.append(a)
        opr.append(b)
    st=np.array(st)
    opr=np.array(opr)
    l_max=max(max(st[:,0]),max(st[:,1]))
    Plr,Prl,Pll,Prr,Amlr,Amrl,Amll,Amrr=[],[],[],[],[],[],[],[]
    for i in range(datanum):
        Elr=0.5*(opr[i][0]+1j*opr[i][3]+1j*opr[i][2]-opr[i][1])
        Erl=0.5*(opr[i][0]-1j*opr[i][3]-1j*opr[i][2]-opr[i][1])
        # Ell=0.5*(opr[i][0]-1j*opr[i][3]+1j*opr[i][2]+opr[i][1])
        # Err=0.5*(opr[i][0]+1j*opr[i][3]-1j*opr[i][2]+opr[i][1])
        plr=math.atan2(Elr.imag,Elr.real)-opr[i][4].real
        prl= math.atan2(Erl.imag,Erl.real)-opr[i][4].real
        # prr = math.atan2(Err.imag, Err.real) - opr[i][4].real
        # pll = math.atan2(Ell.imag, Ell.real) - opr[i][4].real
        Plr.append(fold(plr))
        Prl.append(fold(prl))
        # Pll.append(fold(pll))
        # Prr.append(fold(prr))
        amlr = abs(Elr)
        amrl = abs(Erl)
        # amll = abs(Ell)
        # amrr = abs(Err)
        Amlr.append(amlr)
        Amrl.append(amrl)
        # Amrr.append(amrr)
        # Amll.append(amll)
    Ammax=max(max(Amlr),max(Amrl))
    print(Ammax,l_max)

    with open (pathout+str(wavelength)+'_dataset.txt','a') as f:
        f.truncate(0)
        for i in range(datanum):
            for j in range(2):
                f.write(str('%.4f'%(st[i][j]/l_max))+',')
            f.write(str('%.4f'%(st[i][2]/360+0.5))+','+str('%.4f'%(st[i][3]/180))+',')
            f.write(str('%.4f'%(np.sin(Plr[i])/2+0.5))+','+str('%.4f'%(np.cos(Plr[i])/2+0.5))+',')
            f.write(str('%.4f'%(np.sin(Prl[i])/2+0.5))+','+str('%.4f'%(np.cos(Prl[i])/2+0.5))+',')
            # f.write(str('%.4f'%(np.sin(Pll[i])/2+0.5))+','+str('%.4f'%(np.cos(Pll[i])/2+0.5))+',')
            # f.write(str('%.4f'%(np.sin(Prr[i])/2+0.5))+','+str('%.4f'%(np.cos(Prr[i])/2+0.5))+',')
            f.write(str('%.4f'%(Amlr[i]/Ammax))+','+str('%.4f'%(Amrl[i]/Ammax))+'\n')
            #f.write(str('%.4f'%(Amll[i]/Ammax))+','+str('%.4f'%(Amrr[i]/Ammax))+'\n')
        f.close()
    '''
    plt.figure()
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.scatter(Plr, Prl,s=1,c=Amlr,cmap='RdYlBu_r')
    plt.colorbar()
    plt.show()
    '''
