#-----------------------------可视化数据分布----------------------------
# import torch
# import random
# import matplotlib.pyplot as plt
# import torch

# def analyze_spectrum_per_dimension(spectrum):
#     # 计算每个维度的均值和方差
#     mean_per_dim = torch.mean(spectrum, dim=(0, 2))
#     std_per_dim = torch.std(spectrum, dim=(0, 2))
    
#     print("每个维度的数据分布统计：")
#     for i in range(6):
#         print(f"维度 {i+1}:")
#         print(f"  均值: {mean_per_dim[i]:.4f}")
#         print(f"  标准差: {std_per_dim[i]:.4f}")


# def visualize_spectrum_histogram(spectrum):
#     # 将 spectrum 转换为 numpy 数组以便绘图
#     spectrum_np = spectrum.numpy()
    
#     plt.figure(figsize=(12, 6))
    
#     # 绘制 6 个维度的直方图
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)  # 创建 2x3 的子图布局
#         plt.hist(spectrum_np[:, i, :].flatten(), bins=50)
#         plt.title(f"Spectrum Dimension {i+1} Histogram")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("/data/group_003/yjh/data_bute.png")

# def main():
#     # 动态加载数据
#     spectrum = torch.load('/data/group_003/data_set_b/new_data_spectrum_8000.pt')  # shape: [B, 6, 201]
#     # spectrum = torch.load('/data/group_003/yjh/data_set/tensor_spectrum2000.pt')
#     # 可视化直方图
#     visualize_spectrum_histogram(spectrum[:,1:7,:])
#     # analyze_spectrum_per_dimension(spectrum)

# if __name__ == "__main__":
#     main()



# import torch 
# pat1 = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_3000.pt")
# pat2 = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_6000.pt")

# pat3 = torch.cat((pat1,pat2),dim=0)
# print(pat1.shape)
# print(pat2.shape)
# print(pat3.shape)
# torch.save(pat3,"/data/group_003/yjh/data_set_s/patterns_synthesis_9000.pt")
#-------------------------------------------------------------------
#-----------------随机选取数据-----------------------------------------
import torch
import random
from ipdb import set_trace

# 加载数据
pattern = torch.load('/data/group_003/yjh/data_set/new_pattern.pt')  # shape: [B, 400, 400]
spectrum = torch.load('/data/group_003/yjh/data_set/new_spectrum.pt')  # shape: [B, 6, 201]

print("shape of pattern:", pattern.shape)
print("shape of spectrum:", spectrum.shape)

# 获取总样本数
B = pattern.size(0)

# 检查样本数是否足够
if B < 6000:
    print("Error: The total number of samples is less than 6000. Please use a smaller number or check the dataset.")
else:
    # 随机生成索引
    idx = torch.randperm(B)[:6000]

    # 使用索引切片
    p1 = pattern[idx]
    s1 = spectrum[idx]

    # 保存结果
    torch.save(p1, "6000_real_pattern.pt")
    torch.save(s1, "6000_real_spectrum.pt")
