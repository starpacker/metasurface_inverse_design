import os
import sys
import numpy as np
import time

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

def py_FDTD(model_file,lsf_script,oristr,z,cl):
    file1 = "transmission_data.txt"
    if os.path.exists(file1):
        time.sleep(0.1)
        os.remove(file1)
    copy_and_modify_lsf(lsf_script,oristr,z,cl)
    run_lumerical(lsf_script,model_file)
    return True

def copy_and_modify_lsf(configfile,oristr,z,cl):
    fd=""
    sentence=[]
    for i in range(20):
        t='setnamed("pix'+str(oristr[i])+'","enabled",0);\n'
        sentence.append(t)
    hz="hz="+str(z)+";\n"
    cll="cl="+str(cl)+";\n"
    i=0
    with open(configfile,"r", encoding='utf-8') as f:
        for line in f:
            if 'setnamed("pix' in line:
                line=sentence[i]
                i+=1
            if 'cl=' in line:
                line=cll
            if 'hz=' in line:
                line=hz
            fd+=line
    with open(configfile, "w") as f1:
        f1.write(fd)
    f1.close()
    f.close()
    time.sleep(0.1)

def run_lumerical(lsffile,fspfile):
    foldpath="\"C:\\Program Files\\Lumerical\\FDTD\\bin\\fdtd-solutions.exe\""
    cmd = " ".join([foldpath, fspfile, " -nw -run ", lsffile])
    #" -nw -run "
    os.system(cmd)
    return True

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

if __name__ == '__main__':
    #mater="Au (Gold) - Palik";
    sys.path.append(os.path.dirname(__file__))
    tolnum=10000
    alr=0
    prostrs,prostr,oristr=[],[],[]
    oristr=read_txt("st_test.txt",',','int8')
    for i in range(tolnum-alr):
        i=alr+i
        T=""
        z=0.03
        cyclen=0.4
        py_FDTD("V_stru.fsp","V_structure_correct.lsf",oristr[i],z,cyclen)
        T=read_txt('single_data.txt','\t','str')
        with open("D:\\PengPu\\QR_code\\dataset\\"+str(i)+".txt", "a") as f:
            for j in range(103):
                for k in range(4):
                    f.write(str(T[j][k])+',')
                f.write(str(T[j][4])+'\n')
            for k in range(19):
                f.write(str(oristr[i][k]) + ',')
            f.write(str(oristr[i][19]) + '\n')
            f.close()
        time.sleep(0.1)
        if os.path.exists('single_data.txt'):
            os.remove('single_data.txt')
