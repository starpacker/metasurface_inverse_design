import numpy as np
from pyautocad import Autocad, APoint
if __name__ == '__main__':
    acad=Autocad(create_if_not_exists=True)
    d=0.0001
    p=0.04

    stru='0100100111101111011111000000111101010110000001000110111001000111101111011010111010111101111110111100'
    st = []
    for i in range(len(stru)):
        substring =stru[i:i+1]
        st.append(int(substring))
    xc=0
    yc=0
    Pix=[[0 for i in range(10)] for j in range(10)]

    for i in range(100):
        if st[i]==1:
            y0=i//10
            x0=i%10
            Pix[y0][x0]=1

    D2, U2 = [], []
    for j in range(10):
        x=xc+(j-5)*p
        U,D=[],[]
        if Pix[0][j] == 1:
            D.append(yc-5*p+d)
        for i in range(1,10):
            if Pix[i][j]-Pix[i-1][j]==1:
                D.append(yc+(i-5)*p+d)
            if Pix[i][j]-Pix[i-1][j] ==-1:
                U.append(yc+(i-5)*p-d)
        if Pix[9][j]== 1:
            U.append(yc+5*p-d)

        U1,D1=[],[]
        joi=[0 for i in range(10)]
        for i in range(10):
            if j!=9:
                if Pix[i][j]+Pix[i][j+1]==2:
                    joi[i]=1
        if joi[0]==1:
            D1.append(yc-5*p+d)
        for i in range(1,10):
            if joi[i]-joi[i-1]==1:
                D1.append(yc+(i-5)*p+d)
            if joi[i]-joi[i-1]==-1:
                U1.append(yc+(i-5)*p-d)
        if joi[9]==1:
            U1.append(yc+5*p-d)

        for i in range(len(U)):
            D[i]=float("%.4f"%D[i])
            U[i]=float("%.4f"%U[i])
        for i in range(len(U1)):
            D1[i]=float("%.4f"%D1[i])
            U1[i]=float("%.4f"%U1[i])

        co,Dd,Uu=[],[],[]
        for i in range(10):
            co.append(Pix[i][j])
        for i in range(len(U2)):
            num=int((U2[i]-D2[i])/p)+1
            stn=int((D2[i]-yc)/p+5)
            for k in range(num):
                co[stn+k]=0

        if co[0] == 1:
            Dd.append(yc-5*p+d)
        for k in range(1,10):
            if co[k]-co[k-1]==1:
                Dd.append(yc+(k-5)*p+d)
            if co[k]-co[k-1]==-1:
                Uu.append(yc+(k-5)*p-d)
        if co[9]==1:
            Uu.append(yc+5*p-d)
        for i in range(len(Uu)):
            acad.model.AddLine(APoint(x+d,Dd[i]),APoint(x+d,Uu[i]))  #left

        if j>0:
            U2,D2=[],[]

        # for i in range(len(U)):
        #     for k in range(Ul):
        #         if Ul[k]==U[i]:
        #             acad.model.AddLine(APoint(x-d,D[i]),APoint(x+p-d,D[i]))  #down

        for i in range(len(U)):
            acad.model.AddLine(APoint(x+d,D[i]),APoint(x+p-d,D[i]))#down
            acad.model.AddLine(APoint(x+d,U[i]),APoint(x+p-d,U[i]))#up

            jind=0
            for k in range(len(D1)):
                if jind==0:
                    dr=(D1[k]-U[i])*(D1[k]-D[i])
                    lr=(U1[k]-U[i])*(U1[k]-D[i])
                    if  dr< 0:
                        acad.model.AddLine(APoint(x+p-d,D[i]),APoint(x+p-d,D1[k]))  #right
                        jind+=1
                    if  lr< 0:
                        acad.model.AddLine(APoint(x+p-d,U1[k]),APoint(x+p-d,U[i]))  #right
                        jind+=1
                    if dr==0 and lr==0:
                        jind+=1
                    if jind!=0:
                        D2.append(D1[k])
                        U2.append(U1[k])
            if jind==0:
                acad.model.AddLine(APoint(x+p-d,D[i]),APoint(x+p-d,U[i]))  #right
