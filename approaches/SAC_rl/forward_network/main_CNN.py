import os 
import os.path
import sys
import torch
import torch.utils.data as data
from torch.cuda import amp
import torch.nn as nn
import torch.nn.functional as F
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

from utils import seed_define
from utils import parser_define
from utils import load_data

from train import train_CNN
from sklearn.model_selection import train_test_split


class EPDataset(Dataset):
    def __init__(self,struct, spectrum):
        # 将独立的各个键值合并为一个张量
        # 这里的张量必须有相同的第一维度长度
        self.struct = torch.tensor(struct, dtype=torch.float32)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)
    def __getitem__(self, index):
        return self.struct[index], self.spectrum[index]

    def __len__(self):
        return self.struct.shape[0]


seed_define(seed=3407)
args = parser_define()
print(args)


NetDefine = 'CNN'


args.b = 512   # batch size
args.lr = 0.001
args.out_dir = "/data/group_003/yjh/logs"
args.device = "cuda:1"
start_epoch = 0
end_epoch = 300
criterion = nn.MSELoss()

from MetasurfaceCNN import Meta_CNN_Net
# from CRNN_attention import Meta_CNN_Net
net = Meta_CNN_Net()
net.CNNLayer[0].layer3[0].relu3 = nn.ELU()


net.to(args.device)


min_test_loss = 100000

optimizer = None
if args.opt == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
elif args.opt == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    raise NotImplementedError(args.opt)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# pattern = torch.load("/data/group_003/yjh/data_set/tensor_data_matrix2000.pt")
# spectrum =torch.load("/data/group_003/yjh/data_set/tensor_spectrum2000.pt")
# x_train, x_val, y_train, y_val = train_test_split(pattern, spectrum[:,1:7,:], test_size=0.1)

# pattern = torch.load('combined_pattern.pt')
# spectrum = torch.load('combined_spectrum.pt')

num_batches = 8
pattern = None
for i in range(num_batches):
    filename = f"/data/group_003/data_set_b/new_data_matrix_{i}.pt"
    batch = torch.load(filename)
    if pattern is None:
        pattern = batch
    else:
        pattern = torch.cat((pattern, batch), dim=0)
    print(i+1,pattern.shape)
print(f"Merged array shape: {pattern.shape}")
spectrum = torch.load(f'/data/group_003/data_set_b/new_data_spectrum_{num_batches}000.pt')
print(f"spectrum shape:{spectrum.shape}")


print("shape of pattern:",pattern.shape)
print("shape of spectrum:",spectrum.shape)

x_train, x_val, y_train, y_val = train_test_split(pattern, spectrum[:,1:7,:], test_size=0.1)

train_set = EPDataset(
    struct=x_train,
    spectrum=y_train)

test_set = EPDataset(
    struct=x_val,
    spectrum=y_val)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.b,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.b,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=True
)


# print("Pattern shape:", pattern.shape)  [400,400] 
# print("Spectrum shape:", spectrum.shape)  [7,201]

# print("x_train shape:", x_train.shape) 
# print("y_train shape:", y_train.shape) 
# print("x_val shape:", x_val.shape)      [6,201]
# print("y_val shape:", y_val.shape)      [6,201]


# # 打印第一个样本的输入（struct）和输出（spectrum）
# print("===== 原始数据中的第一个样本 =====")
# print("输入 (struct) 的形状:", pattern[0].shape)


# print('spectrum',spectrum[0])
# print('spectrum',spectrum[0].shape)
# print('spectrum',spectrum[0][0])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if NetDefine == 'CNN':
        out_dir = os.path.join(args.out_dir, f'CNN_8k')
        min_test_loss = 1000000
        # sys.exit()
        weight_changes, learning_rates, net = train_CNN(
            out_dir = out_dir,
            min_test_loss=min_test_loss,
            net = net,
            start_epoch = start_epoch,
            end_epoch = end_epoch,
            criterion = criterion,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            train_data_loader = train_data_loader,
            test_data_loader = test_data_loader,
            args=args)
