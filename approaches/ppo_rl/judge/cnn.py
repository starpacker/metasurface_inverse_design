import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F

# 启用内存优化配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 数据集类（保持不变）
class EPDataset(Dataset):
    def __init__(self, struct, spectrum, transform=None):
        # 展平图像数据为向量 (B, 160000)
        self.struct = torch.tensor(struct, dtype=torch.float32).view(-1, 400*400)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)
        self.transform = transform  # 全连接网络不需要图像变换

    def __getitem__(self, index):
        struct = self.struct[index]
        spectrum = self.spectrum[index]
        return struct, spectrum

    def __len__(self):
        return self.struct.shape[0]

# U-Net 模型定义（修复版）
class UNetRegressor(nn.Module):
    def __init__(self):
        super(UNetRegressor, self).__init__()

        # 编码器部分
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # 中间层
        self.bottleneck = self.conv_block(256, 512)

        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # 最终输出层
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.view(-1, 1, 400, 400)  # 保证输入是 (B, 1, 400, 400)

        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 中间瓶颈层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # 解码器
        dec4 = self.upconv4(bottleneck)
        # 确保尺寸匹配进行拼接
        if dec4.size()[2:] != enc4.size()[2:]:
            enc4 = F.interpolate(enc4, size=dec4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.size()[2:] != enc3.size()[2:]:
            enc3 = F.interpolate(enc3, size=dec3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.size()[2:] != enc2.size()[2:]:
            enc2 = F.interpolate(enc2, size=dec2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.size()[2:] != enc1.size()[2:]:
            enc1 = F.interpolate(enc1, size=dec1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # 最终输出
        out = self.final_conv(dec1)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)  # flatten to (batch_size, 1)
        out = self.regressor(out)
        return out
    
# 其余代码保持不变...
def load_processed_data(output_dir, data_num=2000):
    patterns = []
    responses = []
    count_0 = int(0.4 * data_num)  # CD > 0.0
    count_1 = int(0.2 * data_num)  # CD > 0.2
    count_2 = int(0.3 * data_num)  # CD > 0.3
    count_3 = int(0.1 * data_num)  # CD > 0.5
    for npz_file in glob.glob(os.path.join(output_dir, "*.npz")):
        data = np.load(npz_file)
        # print(data.files)
        if 'response' not in data.files:
            print(f"Warning: 'response' key not found in {npz_file}. Skipping this file.")
            continue # 跳过当前文件
        CD = float(data['response'])
        flag = False
        if 0 < CD <= 0.2 and count_0 >0:
            count_0 -= 1
            flag = True
        if 0.2 < CD <= 0.3 and count_1 >0:
            count_1 -= 1
            flag = True
        if 0.3 < CD <= 0.5 and count_2 >0:
            count_2 -= 1
            flag = True
        if 0.5 < CD < 1 and count_3 >0:
            count_3 -= 1
            flag = True
        if flag:
            patterns.append(data['pattern'])
            responses.append(CD)
        if len(patterns) >= data_num:
            break
    print(count_0,count_1,count_2,count_3)
    print(f'Loaded {len(patterns)} samples')
    return np.array(patterns), np.array(responses)

def load_processed_data_simple(output_dir):
    """
    最简单的数据加载函数
    """
    patterns = []
    responses = []
    for npz_file in glob.glob(os.path.join(output_dir, "*.npz")):
        data = np.load(npz_file)
        patterns.append(data['pattern'])
        responses.append(float(data['response']))
    return np.array(patterns), np.array(responses)

# 修改 load_test_data 为更通用的 load_high_cd_data
def load_high_cd_data(csv_file, max_samples=None):
    """
    从 CSV 文件中读取结构参数和 CD 值，并生成 pattern 图像。
    参数:
        csv_file (str): CSV 文件路径。
        max_samples (int, optional): 最大加载样本数。如果为 None，则加载所有。
    返回:
        tuple: (patterns数组, responses数组)
    """
    df = pd.read_csv(csv_file)
    pra_columns = [col for col in df.columns if col.startswith('para_')]
    pra_list = df[pra_columns].values.tolist()
    cd_list = df['CD'].values.tolist()

    # 如果指定了 max_samples，则只加载前 max_samples 个
    if max_samples is not None:
        pra_list = pra_list[:max_samples]
        cd_list = cd_list[:max_samples]

    patterns = []
    responses = []
    for i, pra in enumerate(pra_list):
        pattern = get_pattern(pra)
        patterns.append(pattern)
        responses.append(cd_list[i])
    X = np.array(patterns)
    y = np.array(responses)
    print(f"Loaded {len(X)} samples from {csv_file}")
    return X, y

# 训练函数（修改输入处理）
def train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler, device, num_epochs=25, writer=None):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            counter = 0
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                counter += 1
                flag = False
                if counter % 50 == 0:
                    flag = True
                    print(counter)
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # if flag:
                    #     print("outputs:", outputs.cpu().detach().numpy())
                    #     print("labels :", labels.cpu().detach().numpy())
                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                        optimizer.step()
                # 统计损失
                running_loss += loss.item() * inputs.size(0)
                if phase == 'val':
                    all_preds.append(outputs.cpu().detach())
                    all_labels.append(labels.cpu().detach())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            # TensorBoard 记录
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                if phase == 'val':
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                    preds = torch.cat(all_preds).numpy()
                    labels = torch.cat(all_labels).numpy()
                    # 记录前10个样本的预测vs实际值
                    for i in range(min(10, len(preds))):
                        writer.add_scalars(f'PredictionsVsLabels/sample_{i}', {
                            'prediction': preds[i].item(),
                            'actual': labels[i].item()
                        }, epoch)
                    # 计算并记录验证集的MAE和RMSE
                    mae = np.mean(np.abs(preds - labels))
                    rmse = np.sqrt(np.mean((preds - labels) ** 2))
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)
            if phase == 'val':
                print("{:<10} {:<15} {:<15} {:<15}".format("样本ID", "预测值", "实际值", "差距"))
                sample_idx = 0
                # 检查 test_loader 是否存在且非空
                if test_loader is not None and len(test_loader.dataset) > 0:
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device).view(-1, 1)
                            outputs = model(inputs)
                            # 遍历批次中的每个样本
                            for i in range(outputs.size(0)):
                                pred = outputs[i].item()
                                actual = labels[i].item()
                                diff = abs(pred - actual)
                                print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                                    sample_idx, pred, actual, diff))
                                sample_idx += 1
                else:
                    print("No test data provided for prediction display.")
                # print('-' * 60 + '\n') # 修复了原始代码中的换行符问题
            # 保存最佳模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"new best model of {best_loss}")
                torch.save(model.state_dict(), 'best_model_unet.pth')
        scheduler.step()
        print()
    print(f'Best Val Loss: {best_loss:.4f}')
    return model

def get_pattern(pra):
    pra1,pra2,pra3,pra4,pra5,pra6,pra7,pra8,pra9, pra10 = pra
    pattern = np.zeros((400, 400), dtype=int)
    for xi in range(pattern.shape[0]):
        for yj in range(pattern.shape[1]):
            if xi >= pra1 and xi <= pra1 + pra3:
                if yj >= pra10 + pra9 + pra6 + pra5 and \
                        yj <= pra10 + pra9 + pra6 + pra5 + pra2:
                    pattern[xi, yj] = 1
            if xi >= pra1 and xi <= pra1 + pra4:
                if yj >= pra10 + pra9 + pra6 and \
                        yj <= pra10 + pra9 + pra6 + pra5:
                    pattern[xi, yj] = 1
            if xi >= pra7 and xi <= pra7 + pra8:
                if yj >= pra10 and yj <= pra10 + pra9:
                    pattern[xi, yj] = 1
    return pattern

# 主程序
if __name__ == '__main__':
    # 超参数
    batch_size = 4  # U-Net 更占显存，建议减小 batch_size
    num_epochs = 500
    learning_rate = 1e-4
    # 创建TensorBoard日志目录
    log_dir = f"runs/unet_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    # 1. 加载原始数据
    pattern_main, spectrum_main = load_processed_data("./dataset2")
    # pattern_main, spectrum_main = load_processed_data_simple("./dataset3")
    print(f"Loaded main  {len(pattern_main)} samples")

    # 2. 加载额外的高 CD 数据
    high_cd_files = [
        "high_cd_data/0.60_cd_samples.csv",
        "high_cd_data/0.75_cd_samples.csv"
    ]
    extra_patterns_list = []
    extra_spectrums_list = []

    for file in high_cd_files:
        if os.path.exists(file): # 检查文件是否存在
             # 可以根据需要调整每个文件加载的最大样本数，或者加载全部(None)
            p, s = load_high_cd_data(file, max_samples=None) # 示例：每个文件加载最多 500 个样本
            extra_patterns_list.append(p)
            extra_spectrums_list.append(s)
        else:
            print(f"Warning: File {file} not found. Skipping.")

    # 3. 合并所有数据
    if extra_patterns_list: # 如果加载到了额外数据
        all_patterns = [pattern_main] + extra_patterns_list
        all_spectrums = [spectrum_main] + extra_spectrums_list
        pattern_combined = np.vstack(all_patterns)
        spectrum_combined = np.hstack(all_spectrums)
        print(f"Combined  {len(pattern_combined)} samples")
    else:
        pattern_combined = pattern_main
        spectrum_combined = spectrum_main
        print("No extra high CD data loaded. Using main data only.")


    # 4. 划分训练集和验证集 (基于合并后的数据)
    x_train, x_val, y_train, y_val = train_test_split(pattern_combined, spectrum_combined, test_size=0.1, random_state=42) # 添加 random_state 保证可复现性

    # 5. 加载用于测试/展示的 0.85 CD 数据 (保持原逻辑)
    test_csv_path = "high_cd_data/0.85_cd_samples.csv"
    if os.path.exists(test_csv_path):
        pattern_test, spectrum_test = load_high_cd_data(test_csv_path, max_samples=1000)
    else:
        print(f"Warning: Test file {test_csv_path} not found. Creating empty test set.")
        pattern_test, spectrum_test = np.array([]), np.array([]) # 创建空的测试集

    # 6. 创建数据集和数据加载器
    image_datasets = {
        'train': EPDataset(x_train, y_train),
        'val': EPDataset(x_val, y_val)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # 为测试集创建 DataLoader (需要检查是否为空)
    if len(pattern_test) > 0 and len(spectrum_test) > 0:
        test_dataset = EPDataset(pattern_test, spectrum_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        test_loader = None # 如果没有测试数据，则设为 None
        print("No test data available for DataLoader.")


    # 模型初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetRegressor().to(device)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 参数初始化
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    
    # 训练模型
    model = train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler, device, num_epochs, writer)
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看训练过程:")
    print(f"tensorboard --logdir=runs")