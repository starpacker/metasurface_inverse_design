import GPyOpt
import numpy as np
from FDTD_api import vec2mat,py_FDTD
import matplotlib.pyplot as plt

spec_num = 61
ini_num = 100
batch_size = 50 # 批处理量,核数,以及迭代次数
num_cores = 2
Dim = 400
loop = 10
run_ini = 1

if __name__ == '__main__':
    
    '''
    按照开关设置是否需要计算初始值
    '''
    if run_ini == 1:
        # 设置初始值 x_ini 和 y_ini
        x_ini = np.random.randint(0,2,(ini_num,Dim))
        y_ini = np.zeros((np.shape(x_ini)[0],1))
        spec_for_ini = np.zeros((np.shape(x_ini)[0],spec_num))
        spec_back_ini = np.zeros((np.shape(x_ini)[0],spec_num))
        for i in range(np.shape(x_ini)[0]):
            stru = vec2mat(x_ini[i,:])
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(stru, cmap ='gray')
            ax.set_title('initial structure'+str(i)), plt.xticks([]), plt.yticks([])
            plt.show()
            spec_for, spec_back = py_FDTD(matrix = stru)
            spec_for_ini[i,:] = spec_for
            spec_back_ini[i,:] = spec_back
            score = spec_for/spec_back
            y_single = np.sum(score)
            y_ini[i] = y_single
        print("ini <" + str(i) + "> = " + str(y_ini))
        print(" ")
        print("ini <" + str(i) + "> MAX =  " + str(max(y_ini))) 
        print(10 * " ##### ")
    
    '''
    设置 BO 自变量参数定义域
    '''
    # 设定 domain 字典
    domain = [{'name': 'var1', 'type': 'discrete', 'domain': (0,1)}]
    for i in range(2,Dim + 1):
        j = i-1
        domainin =[{'name': 'var'+str(i), 'type': 'discrete', 'domain': (0,1)}]
        domain = domain + domainin
        
    '''
    设置BO优化模型, 依据输入的 x y, 产出下一代样本 x 
    '''
    for i in range(1, loop):
        print("Producing samples of generation {}".format(i))
        
        BO_CDmax = GPyOpt.methods.BayesianOptimization(f = None,
                                                    domain = domain,
                                                    model_type = 'GP',
                                                    X = x_ini,
                                                    Y = -y_ini,
                                                    acquisition_type = 'EI',
                                                    evaluator_type = 'local_penalization',
                                                    batch_size = batch_size,
                                                    num_cores = num_cores,
                                                    optimize_restarts = 10,
                                                    de_duplication = True,
                                                    acquisition_jitter = 0.15  # 越大跳的越远?还是越小？
                                                    )
        x_next = BO_CDmax.suggest_next_locations()
        
        '''
        准备对应的 y 样本，并调用fdtd进行计算
        '''
        print("Running FDTD for  {}th   generations".format(i))     
        y_next = np.zeros((batch_size,1))
        spec_for_next = np.zeros((batch_size,spec_num))
        spec_back_next = np.zeros((batch_size,spec_num))
        
        for i1 in range(batch_size):
            stru = vec2mat(x_next[i1,:])
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(stru, cmap ='gray')
            ax.set_title('gene' + str(i) + "sample" + str(i1)), plt.xticks([]), plt.yticks([])
            plt.show()
            spec_for, spec_back = py_FDTD(matrix = stru)
            score = spec_for/spec_back
            spec_for_next[i1,:] = spec_for
            spec_back_next[i1,:] = spec_back
            y_single = np.sum(score)
            y_next[i1] = y_single
        print("gene <" + str(i) + "> = " + str(y_next))
        print(" ")
        print("gene <" + str(i) + "> MAX =  " + str(max(y_next))) 
        print(10 * " ##### ")
        '''
        将新计算得到的x y 样本整合入训练数据，进行下一代计算
        '''
        x_ini = np.concatenate((x_ini, x_next),axis=0)
        y_ini = np.concatenate((y_ini, y_next),axis=0)
        spec_for_ini = np.concatenate((spec_for_ini, spec_for_next),axis=0)
        spec_back_ini = np.concatenate((spec_back_ini, spec_back_next),axis=0)
        
           
