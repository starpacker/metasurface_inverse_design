import math
from keras.models import load_model
from Const import Consts
from utils import load_origin, show_single, my_init
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

print("start")
result = [[] for _ in range(20)]
result_s = [-1e9 for _ in range(20)]
for i in range(Consts["sam_amount"]):
    temp = np.load('result_smaller_mesh_new_data/E2_predict'+ str(i) + '.npy')
    max_s = - 1e9
    max_j = 50
    for j in range(30,60):
        sum1 = 0
        for k in range(1,20):
            if (abs(temp[j][k]) < 0.02) :
                temp[j][k] = (temp[j][k+1] + temp[j][k-1]) / 2
            elif temp[j][k] < - 0.7:
                temp[j][k] = min(temp[j][k], temp[j][k-1])
        for k in range(20):
            sum1 = sum1 + ((20 - k)**0.5) * temp[j][k] 
        if sum1 > max_s:
            max_s = sum1
            max_j = j
    
    print(max_s)
    if result_s[0] > max_s:
        continue
    for k in range(20,999):
            if abs(temp[max_j][k]) < 0.02:
                temp[max_j][k] = (temp[max_j][k+1] + temp[max_j][k-1]) / 2
            elif temp[max_j][k] < - 0.7:
                temp[max_j][k] = min(temp[max_j][k], temp[max_j][k-1])
            else:
                temp[max_j][k] = (temp[max_j][k] + temp[max_j][k-1] + temp[max_j][k+1]) / 3
    for j in range(19):
        if result_s[j+1] > max_s:
            result_s[j] = max_s
            result[j] = temp[max_j]
            break
        else:
            result_s[j] = result_s[j+1]
            result[j] = result[j+1]
            if j == 18:
                result[19] = temp[max_j]
                result_s[19] = max_s

for i in range(20):
    for j in range(Consts["width_mesh"] * Consts["rect_num"]):
        result[i][j] = math.exp(result[i][j]*0.147904034328+0.326595547443)

for i in range(20):
    show_single(result[i], 'best/E2_x'+ str(i), 1000)
                
