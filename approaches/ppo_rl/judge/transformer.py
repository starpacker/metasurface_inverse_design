import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import math

# 启用内存优化配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 参数边界定义
param_bounds = {
    0: (30, 160),
    1: (40, 100),
    2: (40, 80),
    3: (80, 340),
    4: (40, 100),
    5: (40, 100),
    6: (30, 150),
    7: (40, 340),
    8: (40, 100),
    9: (30, 100)
}

class ParameterSpecificNormalizer:
    def __init__(self, param_bounds):
        self.param_bounds = param_bounds
        self.num_params = len(param_bounds)
    
    def normalize(self, params):
        """
        将参数归一化到[0,1]范围
        params: [N, 10] 或 [10]
        """
        params = np.array(params)
        original_shape = params.shape
        if len(original_shape) == 1:
            params = params.reshape(1, -1)
        
        normalized = np.zeros_like(params, dtype=np.float32)
        for i in range(self.num_params):
            min_val, max_val = self.param_bounds[i]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
            # 确保在[0,1]范围内（处理边界情况）
            normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        
        if len(original_shape) == 1:
            return normalized[0]
        return normalized
    
    def denormalize(self, normalized_params):
        """
        将归一化参数还原到原始范围
        normalized_params: [N, 10] 或 [10]
        """
        normalized_params = np.array(normalized_params)
        original_shape = normalized_params.shape
        if len(original_shape) == 1:
            normalized_params = normalized_params.reshape(1, -1)
        
        denormalized = np.zeros_like(normalized_params, dtype=np.float32)
        for i in range(self.num_params):
            min_val, max_val = self.param_bounds[i]
            denormalized[:, i] = normalized_params[:, i] * (max_val - min_val) + min_val
        
        if len(original_shape) == 1:
            return denormalized[0]
        return denormalized

# 创建归一化器实例
normalizer = ParameterSpecificNormalizer(param_bounds)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EdgeLengthTransformerRegressor(nn.Module):
    def __init__(self, seq_len=10, embedding_dim=64, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(EdgeLengthTransformerRegressor, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 连续值嵌入 - 将[0,1]的归一化值映射到高维空间
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.LayerNorm(embedding_dim),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model)  # 添加LayerNorm
        )
        
        # 位置编码（参数位置信息）
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # 参数类型嵌入（区分不同参数的含义）
        self.param_type_embedding = nn.Embedding(seq_len, d_model)
        
        # LayerNorm for input combination
        self.input_layer_norm = nn.LayerNorm(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 - 使用全局池化
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 保留Sigmoid确保输出在[0,1]范围内
        )

    def forward(self, src):
        # src shape: [batch_size, 10] - 每个元素是[0,1]归一化后的值
        # print(src)
        batch_size = src.size(0)
        
        # 添加维度用于嵌入 [batch_size, 10] -> [batch_size, 10, 1]
        values = src.unsqueeze(-1)  # [batch_size, 10, 1]
        
        # 值嵌入
        value_embeddings = self.value_embedding(values)  # [batch_size, 10, d_model]
        
        # 参数类型嵌入
        param_indices = torch.arange(self.seq_len, device=src.device)
        param_embeddings = self.param_type_embedding(param_indices)  # [10, d_model]
        param_embeddings = param_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 10, d_model]
        
        # 合并嵌入并添加LayerNorm
        src = value_embeddings + param_embeddings  # [batch_size, 10, d_model]
        src = self.input_layer_norm(src)  # 添加LayerNorm
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)  # [batch_size, 10, d_model]
        
        # 全局平均池化
        output = output.mean(dim=1)  # [batch_size, d_model]
        
        # 最终预测
        output = self.fc_out(output)  # [batch_size, 1]
        return output

# 修改数据集类
class EPDataset(Dataset):
    def __init__(self, struct, spectrum, normalizer):
        # 归一化输入参数
        normalized_struct = normalizer.normalize(struct)
        self.struct = torch.tensor(normalized_struct, dtype=torch.float32)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)
        self.normalizer = normalizer

    def __getitem__(self, index):
        struct = self.struct[index]      # [10] - 归一化后的float值
        spectrum = self.spectrum[index]
        return struct, spectrum

    def __len__(self):
        return self.struct.shape[0]

# 修改数据加载部分
def load_processed_data(output_dir, normalizer, data_num=2000):
    patterns = []
    responses = []
    count_0 = int(0.4 * data_num)
    count_1 = int(0.2 * data_num)
    count_2 = int(0.3 * data_num)
    count_3 = int(0.1 * data_num)
    
    for npz_file in glob.glob(os.path.join(output_dir, "*.npz")):
        data = np.load(npz_file)
        if 'response' not in data.files:
            print(f"Warning: 'response' key not found in {npz_file}. Skipping this file.")
            continue
        CD = float(data['response'])
        flag = False
        if 0 < CD <= 0.2 and count_0 > 0:
            count_0 -= 1
            flag = True
        if 0.2 < CD <= 0.3 and count_1 > 0:
            count_1 -= 1
            flag = True
        if 0.3 < CD <= 0.5 and count_2 > 0:
            count_2 -= 1
            flag = True
        if 0.5 < CD < 1 and count_3 > 0:
            count_3 -= 1
            flag = True
        if flag:
            filename = os.path.basename(npz_file)
            name_without_ext = filename[:-4]  
            params = list(map(int, name_without_ext.split(',')))
            patterns.append(params)
            responses.append(CD)
        if len(patterns) >= data_num:
            break
    print(count_0, count_1, count_2, count_3)
    print(f'Loaded {len(patterns)} samples')
    return np.array(patterns), np.array(responses)

# 修改high_cd_data加载函数
def load_high_cd_data(csv_file, normalizer, max_samples=None):
    df = pd.read_csv(csv_file)
    pra_columns = [col for col in df.columns if col.startswith('para_')]
    pra_list = df[pra_columns].values.tolist()
    cd_list = df['CD'].values.tolist()

    if max_samples is not None:
        pra_list = pra_list[:max_samples]
        cd_list = cd_list[:max_samples]

    patterns = []
    responses = []
    for i, pra in enumerate(pra_list):
        pattern = np.array(pra)
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
                inputs = inputs.to(device)
                
                labels = labels.to(device).view(-1, 1)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(inputs[0],outputs[0],labels[0])
                    loss = criterion(outputs, labels)
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
                    # 动态学习率调度
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                    
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                    preds = torch.cat(all_preds).numpy()
                    labels = torch.cat(all_labels).numpy()
                    # 记录前10个样本的预测vs实际值
                    # for i in range(min(10, len(preds))):
                    #     print(preds[i].item())
                    #     print(labels[i].item())
                    #     print()
                        # writer.add_scalars(f'PredictionsVsLabels/sample_{i}', {
                        #     'prediction': preds[i].item(),
                        #     'actual': labels[i].item()
                        # }, epoch)
                    # 计算并记录验证集的MAE和RMSE
                    mae = np.mean(np.abs(preds - labels))
                    rmse = np.sqrt(np.mean((preds - labels) ** 2))
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)
                    
                    # 保存最佳模型（只保存一个基于Loss的模型）
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print(f"New best model: {best_loss:.6f}")
                        torch.save(model.state_dict(), 'best_model_trans.pth')
            
            # if phase == 'val':
            #     # print("{:<10} {:<15} {:<15} {:<15}".format("样本ID", "预测值", "实际值", "差距"))
            #     sample_idx = 0
            #     # 检查 test_loader 是否存在且非空
            #     if test_loader is not None and len(test_loader.dataset) > 0:
            #         with torch.no_grad():
            #             for inputs, labels in test_loader:
            #                 inputs = inputs.to(device)
            #                 labels = labels.to(device).view(-1, 1)
            #                 outputs = model(inputs)
            #                 # 遍历批次中的每个样本
            #                 for i in range(outputs.size(0)):
            #                     pred = outputs[i].item()
            #                     actual = labels[i].item()
            #                     diff = abs(pred - actual)
            #                     # print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            #                     #     sample_idx, pred, actual, diff))
            #                     sample_idx += 1
            #     else:
            #         print("No test data provided for prediction display.")
    
    print(f'Best Val Loss: {best_loss:.4f}')
    return model

# 打印样本数据函数
def print_sample_data(patterns, responses, normalizer, num_samples=5):
    print("\n" + "="*60)
    print("SAMPLE DATA ANALYSIS")
    print("="*60)
    
    # 打印原始数据
    print(f"原始参数范围检查 (前{num_samples}个样本):")
    for i in range(min(num_samples, len(patterns))):
        original_params = patterns[i]
        cd_value = responses[i]
        print(f"样本 {i+1}:")
        print(f"  原始参数: {original_params}")
        print(f"  CD值: {cd_value:.6f}")
        
        # 检查参数是否在边界内
        for j, param_val in enumerate(original_params):
            min_val, max_val = param_bounds[j]
            if not (min_val <= param_val <= max_val):
                print(f"  ⚠️  参数{j}超出范围: {param_val} (应为 {min_val}-{max_val})")
    
    # 打印归一化后的数据
    print(f"\n归一化后数据 (前{num_samples}个样本):")
    normalized_patterns = normalizer.normalize(patterns[:num_samples])
    for i in range(len(normalized_patterns)):
        print(f"样本 {i+1}:")
        print(f"  归一化参数: {normalized_patterns[i]}")
        print(f"  CD值: {responses[i]:.6f}")
    
    # 统计信息
    print(f"\n数据集统计信息:")
    print(f"  总样本数: {len(patterns)}")
    print(f"  CD值范围: [{np.min(responses):.6f}, {np.max(responses):.6f}]")
    print(f"  CD值均值: {np.mean(responses):.6f}")
    print(f"  CD值标准差: {np.std(responses):.6f}")
    print("="*60 + "\n")

# 主程序
if __name__ == '__main__':
    # 超参数
    batch_size = 256
    num_epochs = 500
    learning_rate = 1e-3  # 提高初始学习率
    # 创建TensorBoard日志目录
    log_dir = f"runs/transformer_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    # 创建归一化器
    normalizer = ParameterSpecificNormalizer(param_bounds)
    
    # 加载数据时传入归一化器
    pattern_main, spectrum_main = load_processed_data("./dataset", normalizer,data_num = 8000)
    print(f"Loaded main  {len(pattern_main)} samples")

    # 打印样本数据
    print_sample_data(pattern_main, spectrum_main, normalizer)

    # 2. 加载额外的高 CD 数据
    high_cd_files = [
        "high_cd_data/0.60_cd_samples.csv",
        "high_cd_data/0.75_cd_samples.csv"
    ]
    extra_patterns_list = []
    extra_spectrums_list = []

    for file in high_cd_files:
        if os.path.exists(file):
            p, s = load_high_cd_data(file, normalizer, max_samples=None)
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
        
        # 打印合并后的样本数据
        print_sample_data(pattern_combined, spectrum_combined, normalizer)
    else:
        pattern_combined = pattern_main
        spectrum_combined = spectrum_main
        print("No extra high CD data loaded. Using main data only.")

    # 4. 划分训练集和验证集 (基于合并后的数据)
    x_train, x_val, y_train, y_val = train_test_split(pattern_combined, spectrum_combined, test_size=0.1, random_state=42) # 添加 random_state 保证可复现性

    # 5. 加载用于测试/展示的 0.85 CD 数据 (保持原逻辑)
    test_csv_path = "high_cd_data/0.85_cd_samples.csv"
    if os.path.exists(test_csv_path):
        pattern_test, spectrum_test = load_high_cd_data(test_csv_path, normalizer, max_samples=1000)
        # 打印测试集样本数据
        print_sample_data(pattern_test, spectrum_test, normalizer)
    else:
        print(f"Warning: Test file {test_csv_path} not found. Creating empty test set.")
        pattern_test, spectrum_test = np.array([]), np.array([]) # 创建空的测试集

    # 6. 创建数据集和数据加载器
    image_datasets = {
        'train': EPDataset(x_train, y_train, normalizer),
        'val': EPDataset(x_val, y_val, normalizer)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # 为测试集创建 DataLoader (需要检查是否为空)
    if len(pattern_test) > 0 and len(spectrum_test) > 0:
        test_dataset = EPDataset(pattern_test, spectrum_test, normalizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        test_loader = None # 如果没有测试数据，则设为 None
        print("No test data available for DataLoader.")

    # 模型初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EdgeLengthTransformerRegressor(
        seq_len=10,
        embedding_dim=64,
        d_model=128, 
        nhead=8, 
        num_layers=4,
        dropout=0.1
    ).to(device)    

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 动态学习率调度器 - 使用ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

    # 打印模型信息
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*60 + "\n")

    # 训练模型
    model = train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler, device, num_epochs, writer)
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看训练过程:")
    print(f"tensorboard --logdir=runs")