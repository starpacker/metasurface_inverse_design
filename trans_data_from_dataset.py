import torch
import numpy as np
import time

def create_pattern(pra, n_x=400, n_y=400):
    pra1 = pra[:, 0].reshape(-1, 1, 1)
    pra2 = pra[:, 1].reshape(-1, 1, 1)
    pra3 = pra[:, 2].reshape(-1, 1, 1)
    pra4 = pra[:, 3].reshape(-1, 1, 1)
    pra5 = pra[:, 4].reshape(-1, 1, 1)
    pra6 = pra[:, 5].reshape(-1, 1, 1)
    pra7 = pra[:, 6].reshape(-1, 1, 1)
    pra8 = pra[:, 7].reshape(-1, 1, 1)
    pra9 = pra[:, 8].reshape(-1, 1, 1)
    pra10 = pra[:, 9].reshape(-1, 1, 1)
    
    xi = np.arange(n_x).reshape(1, -1, 1)
    yj = np.arange(n_y).reshape(1, 1, -1)
    
    # 各个区域条件
    y1_min = pra10 + pra9 + pra6 + pra5
    y1_max = y1_min + pra2
    mask1 = (xi >= pra1) & (xi <= pra1 + pra3) & (yj >= y1_min) & (yj <= y1_max)
    
    y2_min = pra10 + pra9 + pra6
    y2_max = y2_min + pra5
    mask2 = (xi >= pra1) & (xi <= pra1 + pra4) & (yj >= y2_min) & (yj <= y2_max)
    
    y3_min = pra10
    y3_max = pra10 + pra9
    mask3 = (xi >= pra7) & (xi <= pra7 + pra8) & (yj >= y3_min) & (yj <= y3_max)
    
    # 合并所有区域
    mask_total = mask1 | mask2 | mask3
    patterns = mask_total.astype(int)
    
    # 转置
    patterns = np.transpose(patterns, (0, 2, 1))
    
    return torch.from_numpy(patterns)


# pattern = torch.load("new_pattern.pt")
# print(f"Input pattern shape: {pattern.shape}")
# new_pattern = pattern[:8000]
# print(new_pattern.shape)
# torch.save(new_pattern,"new_pattern_8000_old.pt")


# 主程序
pattern = torch.load("new_pattern.pt")
print(f"Input pattern shape: {pattern.shape}")

pattern_np = pattern.numpy()
batch_size = 1000  # 每批次大小，可以根据实际情况调整
num_batches = len(pattern_np) // batch_size

print(num_batches)

for i in range(num_batches):
    start = i * batch_size
    end = (i + 1) * batch_size
    batch = pattern_np[start:end]
    
    # 生成该批次的模式矩阵
    batch_array = create_pattern(batch)
    
    # 保存为文件
    filename = f"/data/group_003/data_set_sxy/new_data_matrix_{i}.pt"
    torch.save(batch_array, filename)
    print(batch_array.shape)
    print(f"Batch {i} saved to {filename}")
    time.sleep(1)



# torch.save(torch.tensor(array), "new_pattern_8000.pt")
# pattern = torch.load("new_pattern_8000.pt")
# print(f"Input pattern shape: {pattern.shape}")

# pattern = torch.load("new_spectrum.pt")
# print(f"Input pattern shape: {pattern.shape}")

# new_pattern = pattern[:7000]
# print(new_pattern.shape)
# torch.save(new_pattern,"/data/group_003/data_set_sxy/new_spectrum.pt")
