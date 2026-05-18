#读取完成训练的模型，根据拓扑荷生成结构阵列
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
def read_txt(filename):
    file = open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    T = []
    for x in a:
        T.append(x.split(','))
    T = np.array(T)
    T=T.astype('float32')
    return(T)
def fold(x):
    x=float(x)
    pi=np.pi
    for i in range(10):
        if x<=pi and x>-1*pi:
            break
        if x<-1*pi:
            x+=2*pi
        if x>pi:
            x-=2*pi
    return x
class OtS(nn.Module):
    def __init__(self):
        super(OtS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
class StO(nn.Module):
    def __init__(self,dropout=0):
        super(StO, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(64, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(128, 256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.LeakyReLU(),

            nn.Linear(32, 6),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.model(x)
        return x
def val_b_main(path):
    valmodel=torch.load(path)
    testopr=opr
    #准备测试集画图
    testopr=torch.tensor(testopr)
    testopr=testopr.cuda()
    valst = valmodel(testopr)
    valst=valst.detach().cpu().numpy()
    with open (modelpath+"valstnor.txt","a") as f:
        f.truncate(0)
        for i in range(valnum):
            for j in range(3):
                f.write(str('%.4f'%(valst[i][j]))+',')
            f.write(str('%.4f'%(valst[i][3]))+'\n')
def val_f_main(path):
    #premodel.load_state_dict(torch.load(r"D:\PengPu\pythonProject - 0325\model\net2_epoch_109.mdl"))
    valmodel=torch.load(path)
    testst=torch.tensor(st)
    testst=testst.cuda()
    valopr=valmodel(testst)
    valopr=valopr.detach().cpu().numpy()
    with open (modelpath+"valoprnorpre.txt","a")as f:
        f.truncate(0)
        for i in range(valnum):
            for j in range(5):
                f.write(str('%.4f'%(valopr[i][j]))+',')
            f.write(str('%.4f'%(valopr[i][5]))+'\n')
def get_pd(tclr,tcrl,sn,plr0,prl0):
    #sn:side number
    #plr0:-pi~pi
    opr,st,pha=[],[],[]
    for y in range(sn):
        y=sn-y-1
        for x in range(sn):
            r=((y-int(sn/2))**2+(x-int(sn/2))**2)**0.5*0.4 #unit is um
            r=0
            ph_rot=r/75*2*np.pi
            plr=tclr*math.atan2(y-int(sn/2),x-int(sn/2))+ph_rot+plr0 #unit is rad unnormalized
            prl=tcrl*math.atan2(y-int(sn/2),x-int(sn/2))+ph_rot+prl0 #unit is rad unnormalized
            pha.append(np.sin(plr)/2+0.5)
            pha.append(np.cos(plr)/2+0.5)
            pha.append(np.sin(prl)/2+0.5)
            pha.append(np.cos(prl)/2+0.5)
            opr.append(pha)
            pha=[]
    with open(modelpath+'tcphase.txt','a')as f:
        f.truncate(0)
        for i in range(sn**2):
            for j in range(3):
                f.write('%.4f'%(opr[i][j])+',')
            f.write('%.4f'%(opr[i][3])+'\n')
if __name__ == '__main__':
    sn=375
    tclr=-1
    tcrl=1
    plr0=0
    prl0=-0.15
    modelpath='D:\\PengPu\\V\\pythonProject - 0325\\train_val_0701\\bnnpre\\model\\1064nm_E50_3\\'+str(tclr)+'_'+str(tcrl)+'\\'
    get_pd(tclr,tcrl,sn,plr0,prl0)
    valnum=sn**2
    opr=read_txt(modelpath+'tcphase.txt')
    val_b_main(modelpath+"back_epo_1340_loss_13_sc.mdl")
    st=read_txt(modelpath+'valstnor.txt')
    val_f_main(modelpath+"for_epo_1140_loss_29_sincos.mdl")
    T0=read_txt(modelpath+'valoprnorpre.txt')
    T1=read_txt(modelpath+'tcphase.txt')
    plr,plrpre,prl,prlpre,amdif,amave,picx,picy=[],[],[],[],[],[],[],[]

    for i in range(valnum):
        plr.append(math.atan2(T1[i][0]*2-1,T1[i][1]*2-1))
        prl.append(math.atan2(T1[i][2]*2-1,T1[i][3]*2-1))
        plrpre.append(math.atan2(T0[i][0]*2-1,T0[i][1]*2-1))
        prlpre.append(math.atan2(T0[i][2]*2-1,T0[i][3]*2-1))
        Elr=100*(T0[i][4]*0.14327/0.1577)**2
        Erl=100*(T0[i][5]*0.14327/0.1577)**2
        amdif.append(Elr-Erl)
        amave.append((Elr+Erl)/2)

    # for i in range(valnum):
    #    picx.append(fold(plr[i]-plrpre[i]))
    #    picy.append(fold(prl[i]-prlpre[i]))

    plt.figure()
    sl = np.pi
    plt.xlim(-1 * sl, sl)
    plt.ylim(-1 * sl, sl)
    plt.scatter(plr, prl,s=0.1, c='#000000')
    plt.scatter(plrpre, prlpre,s=1, c=amdif, cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('plr/rad')
    plt.ylabel('prl/rad')
    plt.title('Topological charge ('+str(tclr)+','+str(tcrl)+')')
    plt.savefig(modelpath+'\\amdif.jpg')
    plt.show()

    plt.figure()
    sl = np.pi
    plt.xlim(-1*sl,sl)
    plt.ylim(-1*sl,sl)
    plt.scatter(plr,prl,s=0.1, c='#000000')
    plt.scatter(plrpre, prlpre,s=1,c=amave, cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('plr/rad')
    plt.ylabel('prl/rad')
    plt.title('Topological charge ('+str(tclr)+','+str(tcrl)+')')
    plt.savefig(modelpath + '\\amave.jpg')
    plt.show()
