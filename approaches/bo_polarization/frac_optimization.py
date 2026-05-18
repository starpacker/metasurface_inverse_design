# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:54:17 2019

@author: Administrator
"""

import numpy as np
from FDTD_api import py_FDTD,enlarge
from matplotlib import pyplot as plt
import cv2

# first round DBS, with accuracy (4,4); size= np.array([4,4])
def ini_DBS(size, generation):
    index_choose = 30
    score_save_best = np.zeros((1,generation + 1))
    stru_save_best = np.zeros((size[0],size[1],generation + 1))
    score_save = np.zeros((1,generation + 1))
    stru_save = np.zeros((size[0],size[1],generation + 1))
    
    
    initial = np.random.normal(size = size)
    initial[initial>0] = 1
    initial[initial<0] = 0
    forward_ini = py_FDTD(initial)
    score_ini = forward_ini[index_choose]
    forward_save_0 = np.zeros((len(forward_ini),generation + 1))
    forward_save_0[:,0] = forward_ini
    
    print("initial score <" + str(score_ini) + ">")
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(initial, cmap ='gray')
    ax.set_title('initial'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    # save ini spectra
    
    score_save_best[0,0] = score_ini
    stru_save_best[:,:,0] = initial
    score_save[0,0] = score_ini
    stru_save[:,:,0] = initial
    
    for i in range(generation):
        print(15 * "--" + "Runing generation <" + str(i) + ">" + 15 * "--")
        pick_index = (np.random.randint(0,size[0]),np.random.randint(0,size[1]))
        
        # 上一个优化中获得的最优结构
        last_best = np.copy(stru_save_best[:,:,i])
        new_one = stru_save_best[pick_index[0],pick_index[1],i]
        if new_one > 0.5:
            last_best[pick_index] = 0        
        else:
            last_best[pick_index] = 1
        stru_new = last_best[:,:]
        stru_save[:,:,i+1] = stru_new
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(stru_new, cmap ='gray')
        ax.set_title('stru new'), plt.xticks([]), plt.yticks([])
        plt.show()
        
        forward_new= py_FDTD(stru_new)
        forward_save_0[:,i+1] = forward_new
        score_new = forward_new[index_choose]
        score_save[0,i+1] = score_new
        
        
        if score_new > score_save_best[0,i]:
            stru_save_best[:,:,i+1] = stru_new
            score_save_best[0,i+1] = score_new
            print("score improved from <" + str(score_save_best[0,i]) + "> to <" + str(score_new) + ">")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow(stru_save_best[:,:,i+1], cmap ='gray')
            ax.set_title('new best'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            stru_last_best = np.copy(stru_save_best[:,:,i])
            stru_save_best[:,:,i+1] = stru_last_best[:,:]
            score_last_best = np.copy(score_save_best[0,i])
            score_save_best[0,i+1] = score_last_best
            print("<" + str(score_new) + "> no improved, re-search <" + str(score_save_best[0,i+1]) + ">")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow(stru_last_best, cmap ='gray')
            ax.set_title('back to last best'), plt.xticks([]), plt.yticks([])
            plt.show()
        print(41 * "--")
    
    matrix = stru_save_best[:,:,generation]
    score = score_save_best[:,generation]
    return matrix, score, stru_save, score_save, forward_save_0
    
    
'''
确认优化了一个方块以后，需要把可能让他孤立起来的另外点也排除，不能再进行变化
'''   

##########################################

# input matrix, n; output:collection of points location
def findedges(matrix, n):
    matrix2 = enlarge(matrix , matrix2size = np.array([n,n]))
    matrix2 = cv2.copyMakeBorder(matrix2,1,1,1,1,cv2.BORDER_WRAP)
    matrix3 = matrix2[1:n+1, 1:n+1]
    edges_3 = cv2.Laplacian(matrix2,cv2.CV_64F,ksize=1)[1:n+1, 1:n+1]
    plt.subplot(121),plt.imshow(matrix3,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges_3,cmap = 'gray')
    plt.title('Edge Image3'), plt.xticks([]), plt.yticks([])
    plt.show()
    selection = np.vstack(np.nonzero(edges_3))
    return selection


# function input
def level_DBS(matrix, gene, n):
    print("accuracy <" + str(n) + ">")
    selection = findedges(matrix = matrix, n = n)
    size = np.array([n,n])
    generation = gene
    stru_save_best = np.zeros((size[0],size[1],generation + 1))  
    initial = enlarge(matrix,np.array([n,n]))
    forward_ini = py_FDTD(initial)
    forward_save = np.zeros((len(forward_ini),generation + 1))
    score_ini = forward_ini[30]
    score_save_best = np.zeros((1,generation + 1))
    # save ini spectra
    score_save_best[0,0] = score_ini
    stru_save_best[:,:,0] = initial
    
    for i in range(generation):
        print(15 * "--" + "accuracy <" + str(n) + ">" + "generation <" + str(i) + ">" + 15 * "--")
        pick_index = selection[:,np.random.randint(len(selection[1]))]
        last_best = np.copy(stru_save_best[:,:,i])
        new_one = stru_save_best[pick_index[0],pick_index[1],i]
        if new_one > 0.5:
            last_best[pick_index[0],pick_index[1]] = 0        
        else:
            last_best[pick_index[0],pick_index[1]] = 1
        stru_new = last_best[:,:]
        plt.imshow(stru_new, cmap ='gray')
        plt.title('stru new'), plt.xticks([]), plt.yticks([])
        plt.show()
        
        forward_new = py_FDTD(stru_new)
        forward_save[:,i] = forward_new
        score_new = forward_new[30]
        
        if score_new > score_save_best[0,i]:
            stru_save_best[:,:,i+1] = stru_new
            score_save_best[0,i+1] = score_new
            print("score improved from <" + str(score_save_best[0,i]) + "> to <" + str(score_new) + ">")
            plt.imshow(stru_save_best[:,:,i+1], cmap ='gray')
            plt.title('new best'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            stru_last_best = np.copy(stru_save_best[:,:,i])
            stru_save_best[:,:,i+1] = stru_last_best[:,:]
            score_last_best = np.copy(score_save_best[0,i])
            score_save_best[0,i+1] = score_last_best
            print("<" + str(score_new) + "> no improved, re-search <" + str(score_save_best[0,i+1]) + ">")
            plt.imshow(stru_last_best, cmap ='gray')
            plt.title('back to last best'), plt.xticks([]), plt.yticks([])
            plt.show()
            print(41 * "--")
    return stru_save_best,score_save_best,forward_save

re_ini = 4
level_n = 8
gene = 64
matrix_L4 = np.zeros((level_n,level_n,re_ini))
score_L4 = np.zeros((re_ini))
stru_save = np.zeros((level_n,level_n,gene+1, re_ini))
score_save = np.zeros((1,gene+1,re_ini))
'''
此处修改光谱采样点数量后一定需要修改
'''
spec_save = np.zeros((61,gene+1,re_ini))

for i in range(re_ini):
    print(10 * "##" +  "start re-initial DBS <" + str(i) + "> " + 10 * "##")
    matrix_L4[:,:,i], score_L4[i], stru_save[:,:,:,i], score_save[:,:,i], spec_save[:,:,i]= ini_DBS(size = np.array([level_n,level_n]), generation = gene)
socre_max_index = np.argmax(score_L4)
matrix = matrix_L4[:,:,socre_max_index].reshape(level_n,level_n)
              
#stru_save_best_L8,score_save_best_L8,forward_save_L8 = level_DBS(matrix = matrix,gene = 128, n = 8)
#stru_save_best_L16,score_save_best_L16 = level_DBS(matrix = stru_save_best_L8c[:,:,2],gene = 8, n = 16)


################################
