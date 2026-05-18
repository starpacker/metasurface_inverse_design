# ---------------------这里是用来筛选有效光谱激发的数据--------------------------
# import torch
# from ipdb import set_trace
# # 加载数据
# pattern = torch.load('/data/group_003/yjh/data_set/new_pattern.pt')  # shape: [B, 400, 400]
# spectrum = torch.load('/data/group_003/yjh/data_set/new_spectrum.pt')  # shape: [B, 6, 201]

# print("shape of pattern:",pattern.shape)
# print("shape of spectrum:",spectrum.shape)

# # print(spectrum[0][1][0]!=0)
# # 筛选出 spectrum 第一列（第一个通道）的第一个元素不是 0 的数据
# # 注意：spectrum 的第一个通道是 spectrum[:, 0, :]，第一个元素是 spectrum[i][0][0]
# valid_indices = spectrum[:, 1, 0] != 0  # 返回一个布尔张量，指示每个样本是否符合条件

# pattern_filtered = []
# spectrum_filtered = []
# # # 使用布尔索引筛选数据
# for i,x in enumerate(valid_indices):
#     if x == True:
#         pattern_filtered.append(pattern[i])
#         spectrum_filtered.append(spectrum[i])

# pattern_filtered = torch.stack(pattern_filtered,dim=0)
# spectrum_filtered = torch.stack(spectrum_filtered,dim=0)

# print("shape of pattern:",pattern_filtered.shape)
# print("shape of spectrum:",spectrum_filtered.shape)

# # set_trace()
# # 保存筛选后的数据
# torch.save(pattern_filtered, '/data/group_003/yjh/data_set/pattern_filtered.pt')

# torch.save(spectrum_filtered, '/data/group_003/yjh/data_set/spectrum_filtered.pt')


# print("筛选和保存完成！")
#
#---------------------------------------------------------------------

#-------------------------测试每个model的validation loss----------------------------------
import torch
import torch.nn as nn

pattern = torch.load('/data/group_003/yjh/data_set/tensor_data_matrix2000.pt')
spectrum = torch.load('/data/group_003/yjh/data_set/tensor_spectrum2000.pt')

# 定义损失函数
criterion = nn.MSELoss()  # 你可以根据任务选择合适的损失函数，例如 nn.CrossEntropyLoss()
device = "cuda:0"

pattern = pattern.to(device)
spectrum = spectrum.to(device)

# 测试每个模型的损失
def test_model_loss(model, model_path):

    
    # 加载模型权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)
    model.eval()  # 切换到评估模式

    # 计算损失
    with torch.no_grad():
        predictions = model(pattern.unsqueeze(1).float())  # 前向传播
        loss = criterion(predictions, spectrum[:,1:7,:])  # 计算损失
    print(loss.item())
    return loss.item()

# 测试每个模型的损失
def test_model_loss_1(model, model_path):

    
    # 加载模型权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)
    model.eval()  # 切换到评估模式
    trg = torch.randn(2000, 6, 201).to(device)

    # 计算损失
    with torch.no_grad():
        predictions = model(pattern.unsqueeze(1).float(),trg,0)  # 前向传播
        loss = criterion(predictions, spectrum[:,1:7,:])  # 计算损失
    print(loss.item())
    return loss.item()



from MetasurfaceCNN import Meta_CNN_Net
from MetasurfaceCNN import Meta_CRNNAG_Net
net = Meta_CNN_Net()
net.CNNLayer[0].layer3[0].relu3 = nn.ELU()
test_model_loss(net,"/data/group_003/yjh/logs/CNN_8k/checkpoint_max.pth")
test_model_loss(net,"/data/group_003/yjh/logs/CNN_20k/checkpoint_max.pth")

net = Meta_CRNNAG_Net()
test_model_loss_1(net,"/data/group_003/yjh/logs/CRNNAG_v0_8k/checkpoint_max.pth")
test_model_loss_1(net,"/data/group_003/yjh/logs/CRNNAG_v0_20k/checkpoint_max.pth")

from new_agnet import Meta_CRNNAG_Net
net = Meta_CRNNAG_Net()
test_model_loss_1(net,"/data/group_003/yjh/logs/CRNNAG_attention_v0_20k/checkpoint_max.pth")
test_model_loss_1(net,"/data/group_003/yjh/logs/CRNNAG_attention_v0_20k/checkpoint_max.pth")



