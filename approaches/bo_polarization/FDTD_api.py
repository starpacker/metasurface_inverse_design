import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

def py_FDTD(matrix, model_file = "FDTD_model.fsp",mother_script = "FDTD_script.lsf"):
    # make sure the out-dated files are deleted
    file1 = "stru.txt"
    file2 = "forward.txt"
    file3 = "backward.txt"
    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    if os.path.exists(file3):
        os.remove(file3)
    # use matrix to generate stru.txt file for FDTD simulation
    stru = enlarge(matrix)
    np.savetxt("stru.txt",stru)
    copy_and_modify_lsf(mother_script)
    run_lumerical("script_copy.lsf", curmodel=model_file)
    # read txt file
    forward = np.loadtxt("forward.txt")
    backward = np.loadtxt("backward.txt")
    return forward, backward

def copy_and_modify_lsf(configfile):
    with open(configfile,"r", encoding='utf-8') as f:
        lines=f.readlines()
    with open("script_copy.lsf", "w") as f1:
        f1.write("".join(lines))
        
def run_lumerical(configfile, curmodel):
    # Some system specific parameters.
    # Depends on the installation path of Lumerical FDTD
    fdtdsolutions = "\"C:\\Program Files\\Lumerical\\FDTD\\bin\\fdtd-solutions.exe\""
    fspfile = curmodel
    lsffile=configfile
    cmd = " ".join([fdtdsolutions, fspfile, " -nw -run ", lsffile])
    os.system(cmd)
    return 1

def enlarge(matrix1 = np.array([[0,0],[1,0]]), matrix2size = np.array([400,400])):
    matrix2 = np.zeros((matrix2size[0],matrix2size[1]))
    matrix1size = np.array([np.shape(matrix1)[0],np.shape(matrix1)[1]])
    kernelsize = (matrix2size/matrix1size).astype(int)
    for d1 in range(matrix1size[0]):
        for d2 in range(matrix1size[1]):
            core = matrix1[d1,d2]
            flower = core * np.ones((kernelsize[0],kernelsize[1]))
            matrix2[(d1 * kernelsize[0]):((d1+1) * kernelsize[0]),(d2 * kernelsize[1]):((d2+1) * kernelsize[1])] = flower
    return matrix2

# 寻找得到下一精度水平可以变化的左右像素坐标
def level2select(matrix0, n = 8):
    matrix2 = enlarge(matrix1 = matrix0,matrix2size = np.array([n,n]))
    matrix2 = cv2.copyMakeBorder(matrix2,1,1,1,1,cv2.BORDER_WRAP)
    edges_3 = cv2.Laplacian(matrix2,cv2.CV_64F,ksize=1)[1:n+1, 1:n+1]
    plt.imshow(edges_3,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()    
    # use edges_3 for edge manipulation
    selection = np.vstack(np.nonzero(edges_3))
    return selection

'''
1 生成[0,2^core]之间的整数，转化为2进制码，并reshape到(sqrt(core),squrt(core))的矩阵
2 按照 sqrt(dim)* sqrt(dim) = dim 维度进行排列这些数据
'''     
def vec2mat(vector, core = 1):
    dim = np.shape(vector)[0]
    vector = vector.reshape(-1,1)
    dim_s = int(np.sqrt(dim))
    core_s = int(np.sqrt(core))
    # 遍历每一个十进制维度
    matrix = np.zeros([dim_s*core_s,dim_s*core_s])
    for i in range(dim_s):
        for j in range(dim_s):
        # 选出十进制数转化为 2 进制，删除ob mark并且往前填充0到16维
            atom_10 = int(vector[dim_s*i + j])
            atom_2 = bin(atom_10).lstrip("0b")
            atom_normed2 = atom_2.zfill(core)
            atom_normed2 = np.array(list(atom_normed2), dtype=int)
            atom_normed2 = atom_normed2.reshape(core_s,core_s)
            matrix[core_s*i:core_s*i+core_s,core_s*j:core_s*j+core_s] = atom_normed2
    return matrix

'''
程序测试部分
a = np.array([2,1,4,6]).reshape(-1,1)
b = vec2mat(a)

spec_a, spec_b = py_FDTD(matrix = b)
'''
