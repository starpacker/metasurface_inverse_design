#读取完成训练的模型，根据拓扑荷生成结构阵列
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
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
    with open (datapath+"stnor.txt","a") as f:
        f.truncate(0)
        for i in range(16):
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
    with open (datapath+"oprnor.txt","a")as f:
        f.truncate(0)
        for i in range(37):
            for j in range(5):
                f.write(str('%.4f'%(valopr[i][j]))+',')
            f.write(str('%.4f'%(valopr[i][5]))+'\n')
def get_pd():
    tarop,t=[],[]
    for i in range(4):
        for j in range(4):
            t.append(np.sin((j-2)*np.pi/2)*0.5+0.5)
            t.append(np.cos((j-2)*np.pi/2)*0.5+0.5)
            t.append(np.sin((i-2)*np.pi/2)*0.5+0.5)
            t.append(np.cos((i-2)*np.pi/2)*0.5+0.5)
            tarop.append(t)
            t=[]
    with open (datapath+'phase target.txt','a') as f:
        f.truncate(0)
        for i in range(16):
            for j in range(3):
                f.write('%.4f'%(tarop[i][j])+',')
            f.write('%.4f'%(tarop[i][3])+'\n')
    return tarop
def get_st():
    tarst,t=[],[]
    for i in range(37):
        t.append(0.75)
        t.append(0.3333)
        t.append(0.1/0.4349)
        t.append(i*0.01/0.4349)
        tarst.append(t)
        t=[]
    with open (datapath+'st target.txt','a') as f:
        f.truncate(0)
        for i in range(37):
            for j in range(3):
                f.write('%.4f'%(tarst[i][j])+',')
            f.write('%.4f'%(tarst[i][3])+'\n')
    return tarst
def stdenor():
    stn=read_txt(datapath + 'stnor.txt')  # structure normolized
    std=[]  # structure denormolized
    for i in range(len(stn)):
        x = stn[i]
        l1 = x[0] * 0.4349
        l2 = x[1] * 0.4349
        x[2] = (x[2] - 0.5) * 360
        x[3] = x[3] * 180
        a1 = (x[2] - 0.5 * x[3]) / 180 * np.pi
        a2 = (x[2] + 0.5 * x[3]) / 180 * np.pi
        x1 = l1 * np.cos(a1)
        y1 = l1 * np.sin(a1)
        x2 = l2 * np.cos(a2)
        y2 = l2 * np.sin(a2)
        xc = (max(x1, x2, 0) + min(x1, x2, 0)) / 2
        yc = (max(y1, y2, 0) + min(y1, y2, 0)) / 2
        t = []
        t.append(x1 - xc)
        t.append(y1 - yc)
        t.append(x2 - xc)
        t.append(y2 - yc)
        t.append(-xc)
        t.append(-yc)
        std.append(t)
    with open(datapath + 'stxy.txt', 'a')as f:
        f.truncate(0)
        for i in range(len(std)):
            for j in range(5):
                f.write('%.6f' % (std[i][j]) + ',')
            f.write('%.6f' % (std[i][5]) + '\n')
        f.close()
if __name__ == '__main__':
    modelpath='D:\\PengPu\\V\\pythonProject - 0325\\train_val_0701\\bnnpre\\model\\1550nm_1w_loss43\\'
    datapath='D:\\PengPu\\metalens\\st-op\\L-op\\'
    get_st()
    #opr=read_txt(datapath+'opr target.txt')
    #val_b_main(modelpath+"back_epo_1150_loss_36_sc.mdl")
    st=read_txt(datapath+'st target.txt')
    val_f_main(modelpath+"for_epo_1010_loss_33_sincos.mdl")
    #stdenor()
