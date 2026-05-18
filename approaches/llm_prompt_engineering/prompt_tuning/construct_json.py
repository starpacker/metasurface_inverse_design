import json
import numpy as np
import torch

wave_length = 650
def transform_matrix(matrix):
    sum_array = matrix.sum(dim=1)
    flag = True
    for i in range(400):
        if sum_array[i] != 0 and flag:
            b = i
            flag = False
            continue
        if sum_array[i] == 0 and not flag:
            d = i - b
            break
    flag = True
    for i in range(b+d,400):
        if sum_array[i] != 0 and flag:
            f = i
            flag = False
            continue
        if sum_array[i] == 0 and not flag:
            h = i - f
            break
    flag = True
    for i in range(400):
        if matrix[b][i] != 0 and flag:
            a = i
            flag = False
            continue
        if matrix[b][i] == 0 and not flag:
            c = i - a
            flag = True
            break
    flag = True
    for i in range(400):
        if matrix[f][i] != 0 and flag:
            e = i
            flag = False
            continue
        if matrix[f][i] == 0 and not flag:
            g = i - e
            break
    return [a,b,c,d,e,f,g,h]

num_samples = 50
pattern = torch.load('/data/group_003/yjh/combined_pattern.pt')
spectrum = torch.load('/data/group_003/yjh/combined_spectrum.pt')
idx = (wave_length-600) * 2
spectrum = spectrum[:,:,idx]
spectrum_list = spectrum.tolist()
# pattern_list = pattern.tolist()

# 构造带物理注释的数据集
meta_surface_data = []
for i in range(num_samples):
    sample = {
        "index": i,
        "structure": transform_matrix(pattern[i]),
        "wavelength": f"{wave_length}nm",
        "spectrum_response": {
        "r_rl": {"real": round(spectrum_list[i][1],5), "imag": round(spectrum_list[i][2],5)},
        "r_lr": {"real": round(spectrum_list[i][3],5), "imag": round(spectrum_list[i][4],5)},
        "r_rr": {"real": round(spectrum_list[i][5],5), "imag": round(spectrum_list[i][6],5)}
            }
        }
    
    meta_surface_data.append(sample)
    


# 保存JSON文件
with open(f"./meta_surface_dataset_{wave_length}.json", "w") as json_file:
    json.dump(meta_surface_data, json_file, indent=None)

print("JSON successfully generated with enhanced physical descriptions.")



