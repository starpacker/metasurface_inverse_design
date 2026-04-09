import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from matplotlib import pyplot as plt
def read_txt(filename):
    file = open(filename)
    a=file.read()
    a=a.split('\n')
    a.pop()
    for i in range(len(a)):
        a[i]=a[i].split(',')
    a = np.array(a)
    a = a.astype('float32')
    return(a)
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
def get_data_loader(n):
    datast,dataopr=[],[]
    for i in n:
        datast.append(st[i])
        dataopr.append(opr[i])
    return datast,dataopr
def train_main():
    batch_size=300
    learning_rate=1e-3
    epochs = 2000

    #清空记录loss的文档
    with open(path+str(alldatanum)+" loss.txt", "a") as f:
        f.truncate(0)
    f.close()

    device = torch.device('cuda:0')
    net2 = StO().to(device)
    optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
    criteon = nn.MSELoss().to(device)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net2.parameters())))

    total_id=[j for j in range(alldatanum)]
    np.random.shuffle(total_id)
    # 获取训练数据
    trainst,trainopr=get_data_loader([total_id[j] for j in range(int(alldatanum*0.9))])
    trainnum=len(trainst)
    tdi=[j for j in range(trainnum)]
    # 获取测试集
    testnum = alldatanum - trainnum
    testst, testopr = get_data_loader([total_id[j+trainnum] for j in range(testnum)])
    #准备测试集画图
    picid=0#画图物理量在矢量中的序号
    picx, picy = [], []
    for i in range(testnum):
        picx.append(testopr[i][picid])

    #获得batch的数量记为k
    batchst,batchopr=[],[]
    lastbatch=0
    if trainnum%batch_size!=0:
        lastbatch=1
    k=trainnum//batch_size+lastbatch

    telmin = 500
    epochmin = 0
    for epoch in range(epochs):
        losstem=0
        #按照k个batch的格式随机划分训练集，随机很重要
        np.random.shuffle(tdi)
        net2.train()#训练开始
        for i in range(k):
            if lastbatch == 1 and i == k-1:
                bs=trainnum%batch_size
            else:
                bs=batch_size
            for j in range(bs):
                batchst.append(trainst[tdi[i*bs+j]])
                batchopr.append(trainopr[tdi[i*bs+j]])
            batchopr=torch.tensor(batchopr)
            batchst=torch.tensor(batchst)
            batchst = batchst.to(device)
            batchopr = batchopr.cuda()
            logits = net2(batchst)

            loss = criteon(logits, batchopr)
            losstem+=float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchst, batchopr = [], []
        net2.eval()#训练结束

        if epoch % 10 == 0:#每间隔10个epoch做test
            #trl=int((lossfracsum/k)**0.5*1e3)
            #trl0=int((lossfracsum/k)**0.5*1e3)
            trl=int((losstem/k)**0.5*1e3)#获得training loss
            tempt=[]

            plt.figure()
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            if picid==0:
                tag='sin(plr)'
            elif picid==1:
                tag='cos(plr)'
            elif picid==2:
                tag='sin(prl)'
            elif picid==3:
                tag='cos(prl)'
            elif picid==4:
                tag='amlr'
            elif picid==5:
                tag='amrl'
            plt.xlabel('test: '+tag)
            plt.ylabel('test: '+tag+' plrsin predict')

            # 训练集走一遍网络以获得test loss
            plt_st = torch.tensor(testst)
            plt_st = plt_st.to('cuda:0')
            plt_opr = torch.tensor(testopr)
            plt_opr = plt_opr.cuda()
            logits = net2(plt_st)
            #tel0= criteon(logits[:,0], plt_st[:,0])
            tel= criteon(logits, plt_opr)
            tel= int(tel**0.5*1e3)
            #tel0=int(tel0**0.5*1e3)

            #准备画图
            y = logits.detach().cpu().numpy()
            picy=[]
            for i in range(testnum):
                picy.append(y[i][picid])
            plt.scatter(picx, picy)
            plt.title("epoch " + str(epoch) + ", loss " + str('%.4f'%(tel*1e-3)))

            #记录training loss和test loss
            with open(path+str(alldatanum)+" loss.txt", "a") as f:
                f.write(str(trl)+","+str(tel)+"\n")
            f.close()
            print("epoch ",epoch," averange training loss is ",trl,", averange testing loss is ", tel, " *10^(-3)")
            if epoch == 0:
                torch.save(net2, path+str(alldatanum)+'fe' + str(epochmin)+'l'+str(telmin)+'.mdl')
                #plt.savefig(path+'for_epo_' + str(epochmin) + '_loss_' + str(telmin) + '_sincos.jpg')
            if telmin>tel:
                #os.remove(path+'for_epo_' + str(epochmin) + '_loss_' + str(telmin) + '_sincos.jpg')
                os.remove(path+str(alldatanum)+'fe' + str(epochmin)+'l'+str(telmin)+'.mdl')
                epochmin=epoch
                telmin=tel
                #plt.savefig(path+'for_epo_' + str(epochmin) + '_loss_' + str(telmin) + '_sincos.jpg')
                torch.save(net2,path+str(alldatanum)+'fe' + str(epochmin)+'l'+str(telmin)+'.mdl')
            plt.close()
            #plt.show()
if __name__ == '__main__':
    wavelength=1064
    # l1,l2,a,b,plrsin,plrcos,prlsin,prlcos,amlr,amrl
    st, opr = [], []
    path = 'D:\\PengPu\\V\\dataset_num\\'
    T0=read_txt(path+str(wavelength)+'_dataset.txt')
    alldatanum=9000
    T=T0[:alldatanum,:]
    T=T.astype('float32')
    for x in T:
        st.append(x[:4])
        opr.append(x[4:])
    train_main()
