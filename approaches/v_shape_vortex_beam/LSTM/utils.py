from Const import Consts
import numpy as np
import math
import matplotlib.pyplot as plt
from keras import backend as K

def load():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["width_mesh"]*Consts["rect_num"]], dtype=np.float32)
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            for k in range(Consts["rect_num"]):
                l = 0
                while (l <= j) and (l*Consts["mesh_y"] <= tmp[i][k]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 1.5
                    l+=1
                for l in range(j+1, Consts["length_mesh"]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 0
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    for i in range(Consts["sam_amount"]*Consts["length_mesh"]):
        for j in range(Consts["width_mesh"]*Consts["rect_num"]):
            output[i][j] = math.log(output[i][j])
    return input,output

def load_conv():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["width_mesh"]*Consts["rect_num"]], dtype=np.float32)
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            for k in range(Consts["rect_num"]):
                l = 0
                while (l <= j) and (l*Consts["mesh_y"] <= tmp[i][k]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 1.5
                    l+=1
                for l in range(j+1, Consts["length_mesh"]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 0
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    for i in range(Consts["sam_amount"]*Consts["length_mesh"]):
        for j in range(Consts["width_mesh"]*Consts["rect_num"]):
            output[i][j] = math.log(output[i][j])
    input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["width_mesh"]*Consts["rect_num"], 1, 1])
    output = np.reshape(output,
                        [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"], 1, 1])
    return input,output

def load_nodouble():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"]*Consts["rect_num"]], dtype=np.float32)
    for i in range(Consts["sam_amount"]):
        for k in range(Consts["rect_num"]):
            l = 0
            while (l * Consts["mesh_y"] <= tmp[i][k]):
                for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                    input[i][l][p] = 1.5
                l += 1

    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    # output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])

    for i in range(Consts["sam_amount"]):
        for k in range(Consts["length_mesh"]):
            for j in range(Consts["width_mesh"] * Consts["rect_num"]):
                output[i][k][j] = math.log(output[i][k][j])
    return input,output

def load_origin():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["width_mesh"]*Consts["rect_num"]], dtype=np.float16)
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            for k in range(Consts["rect_num"]):
                l = 0
                while (l <= j) and (l*Consts["mesh_y"] <= tmp[i][k]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 1.5
                    l+=1
                for l in range(j+1, Consts["length_mesh"]):
                    for p in range(k * Consts["width_mesh"], (k + 1) * Consts["width_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][p] = 0
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"], dtype=np.float32)
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    return input,output

def load1():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["rect_num"]])
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            for k in range(Consts["rect_num"]):
                l = 0
                while (l <= j) and (l*Consts["mesh_y"] <= tmp[i][k]):
                    input[i * Consts["length_mesh"] + j][l][k] = 1.5
                    l += 1
                for l in range(j+1, Consts["length_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][k] = 0
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    for i in range(Consts["sam_amount"]*Consts["length_mesh"]):
        for j in range(Consts["width_mesh"]*Consts["rect_num"]):
            output[i][j] = math.log(output[i][j])
    return input,output

def load1_origin():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = np.ones([Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"], Consts["rect_num"]])
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            for k in range(Consts["rect_num"]):
                l = 0
                while (l <= j) and (l*Consts["mesh_y"] <= tmp[i][k]):
                    input[i * Consts["length_mesh"] + j][l][k] = 1.5
                    l += 1
                for l in range(j+1, Consts["length_mesh"]):
                        input[i * Consts["length_mesh"] + j][l][k] = 0
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    return input,output


def show(y1, y2, name, n):
    x = np.linspace(0, n, n)
    plt.plot(x, y1, label='origin', linewidth=0.5)
    plt.plot(x, y2, label='predict', linewidth=0.5)
    plt.title("intensity")
    plt.legend()
    plt.savefig(name+".png")
    plt.clf()
    print(name)

def show_single(y1, name, n):
    x = np.linspace(0, n, n)
    plt.plot(x, y1, label='E', linewidth=0.5)
    plt.title("intensity")
    plt.legend()
    plt.savefig(name+".png")
    plt.clf()
    print(name)

def my_init(shape, dtype=None):
    res = K.constant(0, shape=shape, dtype=dtype)
    if len(shape) == 2:
        if 4 * shape[0] == shape[1]:
            for i in range(shape[0]):
                res[i][i] = 1
                res[i][shape[0]+i] = 1
                res[i][shape[0]*2 + i] = 1
                res[i][shape[0]*3 + i] = 1
    return res

def normalization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    return (data - mu) / sigma


def new_load():
    tmp = np.loadtxt(Consts["input_pos"])
    tmp = np.reshape(tmp,[Consts["sam_amount"], Consts["rect_num"]])

    input = [[] for _ in range(Consts["sam_amount"] * Consts["length_mesh"])]
    for i in range(Consts["sam_amount"]):
        for j in range(Consts["length_mesh"]):
            tmp2 = [0 for _ in range(Consts["width_mesh"] * Consts["rect_num"])]
            for k in range(Consts["rect_num"]):
                if j * Consts["mesh_y"] < tmp[i][k]:
                    for l in range(k * Consts["width_mesh"], (k+1) * Consts["width_mesh"]):
                        tmp2[l] = 1
            for k in range(j, Consts["length_mesh"]):
                input[i * Consts["sam_amount"] + k].append(tmp2)
    # input = np.reshape(input, [Consts["sam_amount"] * Consts["length_mesh"], Consts["length_mesh"] * Consts["width_mesh"]*Consts["rect_num"]])
    output = np.loadtxt(Consts["input_E2"])
    output = np.reshape(output, [Consts["sam_amount"], Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    output = np.reshape(output, [Consts["sam_amount"] * Consts["length_mesh"], Consts["width_mesh"] * Consts["rect_num"]])
    for i in range(Consts["sam_amount"]*Consts["length_mesh"]):
        for j in range(Consts["width_mesh"]*Consts["rect_num"]):
            output[i][j] = math.log(output[i][j])
    return input,output
