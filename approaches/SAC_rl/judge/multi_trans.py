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
import gc

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
    def __init__(self, seq_len=10, embedding_dim=64, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, num_outputs=2):
        super(EdgeLengthTransformerRegressor, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_outputs = num_outputs  # 输出两个CD值
        
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
        
        # 输出层 - 使用全局池化，输出两个CD值
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_outputs),  # 修改为输出num_outputs个值
        )

    def forward(self, src):
        # src shape: [batch_size, 10] - 每个元素是[0,1]归一化后的值
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
        
        # 最终预测 - 输出两个CD值
        output = self.fc_out(output)  # [batch_size, 2]
        return output

# 修改数据集类以适应两个输出
class EPDataset(Dataset):
    def __init__(self, struct, spectrum, normalizer):
        # 归一化输入参数
        normalized_struct = normalizer.normalize(struct)
        self.struct = torch.tensor(normalized_struct, dtype=torch.float32)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)  
        self.normalizer = normalizer

    def __getitem__(self, index):
        struct = self.struct[index]      
        spectrum = self.spectrum[index]  
        return struct, spectrum

    def __len__(self):
        return self.struct.shape[0]

def load_data(input_file='pra_reward_dataset.txt'):
    """
    加载保存的PRA-reward数据集
    
    Parameters:
    - input_file: 输入文件名
    
    Returns:
    - pra_list: PRA数据列表
    - reward_list: 奖励值列表
    """
    pra_list = []
    reward_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
            
        values = line.split()
        reward = float(values[0])
        pra_data = np.array([float(val) for val in values[1:]])
        
        # print(pra_data,reward)

        pra_list.append(pra_data)
        reward_list.append(reward)
    
    return pra_list, reward_list

# 修改训练函数以适应两个输出
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, writer=None):
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
                labels = labels.to(device)  # 现在是 [batch_size, 1]

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # [batch_size, 1]
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
                    # 计算并记录验证集的MAE和RMSE（对两个CD值分别计算）
                    mae = np.mean(np.abs(preds[:] - labels[:]))
                    rmse = np.sqrt(np.mean((preds[:] - labels[:]) ** 2))
                    
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)
                    
                    # 保存最佳模型（只保存一个基于Loss的模型）
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print(f"New best model: {best_loss:.6f}")
                        torch.save(model.state_dict(), 'best_model_trans.pth')
            
    print(f'Best Val Loss: {best_loss:.4f}')
    return model


# 主程序
if __name__ == '__main__':
    # 超参数
    batch_size = 256
    num_epochs = 500
    learning_rate = 1e-4
    # 创建TensorBoard日志目录
    log_dir = f"runs/transformer_experiment_multi_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    # 创建归一化器
    normalizer = ParameterSpecificNormalizer(param_bounds)
    
    print("Loading main data...")
    pattern_main, reward_main = load_data(input_file='pra_reward_dataset_10k.txt')
    print(f"Loaded main {len(pattern_main)} samples ")


    # 直接使用8-2划分
    x_train, x_val, y_train, y_val = train_test_split(pattern_main, reward_main, test_size=0.2, random_state=42)

    # 打印划分后的信息
    print(f"\n数据集划分信息:")
    print(f"  训练集样本数: {len(x_train)}")
    print(f"  验证集样本数: {len(x_val)}")

    # 创建数据集和数据加载器
    image_datasets = {
        'train': EPDataset(x_train, y_train, normalizer),
        'val': EPDataset(x_val, y_val, normalizer)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # 模型初始化 - 修改为输出2个值
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EdgeLengthTransformerRegressor(
        seq_len=10,
        embedding_dim=64,
        d_model=128, 
        nhead=8, 
        num_layers=4,
        dropout=0.1,
        num_outputs=1  
    ).to(device)    

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 动态学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

    # 打印模型信息
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Output dimensions: {model.num_outputs} (CD1 and CD2)")
    print("="*60 + "\n")

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, writer)
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看训练过程:")
    print(f"tensorboard --logdir=runs")