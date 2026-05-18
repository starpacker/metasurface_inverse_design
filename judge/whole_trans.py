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
    def __init__(self, seq_len=10, embedding_dim=64, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, num_outputs=201):
        super(EdgeLengthTransformerRegressor, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_outputs = num_outputs  # 输出序列长度为201
        
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
        
        # 输出层 - 使用全局池化，输出201个m值
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),  # 修改为输出num_outputs个值
            nn.Sigmoid()  # 保留Sigmoid确保输出在[0,1]范围内
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
        
        # 最终预测 - 输出201个m值
        output = self.fc_out(output)  # [batch_size, 201]
        return output

# 修改数据集类以适应序列输出
class EPDataset(Dataset):
    def __init__(self, struct, spectrum, normalizer):
        # 归一化输入参数
        normalized_struct = normalizer.normalize(struct)
        self.struct = torch.tensor(normalized_struct, dtype=torch.float32)
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)  # 现在是 [N, 201]
        self.normalizer = normalizer

    def __getitem__(self, index):
        struct = self.struct[index]      # [10] - 归一化后的float值
        spectrum = self.spectrum[index]  # [201] - 201个m值
        return struct, spectrum

    def __len__(self):
        return self.struct.shape[0]

# 新的加载数据函数，支持完整序列
def load_cd_data(data_path='C:/data/pra', max_samples=10000):
    pra_list = []
    cd_list = []

    file_paths = list(glob.iglob(os.path.join(data_path, "*.txt")))
    # 随机打乱文件路径
    import random
    random.shuffle(file_paths)

    count = 0

    for file_path in file_paths:
        if count % 1000 == 0:
            print(count)
        if count >= max_samples:
            break

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pra = list(map(int, base_name.split(',')))

        op = np.loadtxt(file_path)

        # 只处理正常段（op[-1,1] <= 750）
        if op[-1, 1] > 750:
            continue
        
        # 提取所有需要的索引数据并转换为numpy数组
        eig_real_1 = op[:, 18]  # shape: (201,)
        eig_imag_1 = op[:, 19]
        eig_real_2 = op[:, 20]
        eig_imag_2 = op[:, 21]
        
        # 向量化计算
        dis_real = np.abs(eig_real_1 - eig_real_2)
        dis_imag = np.abs(eig_imag_1 - eig_imag_2)
        m_response = (dis_real + dis_imag) / 2  # shape: (201,)

        # r_lr_real = op[:, 10]
        # r_lr_imag = op[:, 11]
        # r_rl_real = op[:, 12]
        # r_rl_imag = op[:, 13]
        # r_rr_real = op[:, 14]
        # r_rr_imag = op[:, 15]

        # r_lr = r_lr_real ** 2 + r_lr_imag ** 2
        # r_rl = r_rl_real ** 2 + r_rl_imag ** 2
        # r_rr = r_rr_real ** 2 + r_rr_imag ** 2

        # m_response = np.abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr)  # cd

        if len(m_response) != 201:
            print(op[-1,1],len(m_response))
            continue
        pra_list.append(pra)
        cd_list.append(m_response)
        count += 1

    del file_paths
    gc.collect()
    print(f'Loaded {len(pra_list)} samples with sequence length {len(cd_list[0])}')
    return np.array(pra_list), np.array(cd_list)

# 修改训练函数以适应序列输出
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
                labels = labels.to(device)  # 现在是 [batch_size, 201]

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # [batch_size, 201]

                    loss = criterion(outputs, labels).mean()  # 对序列求平均损失

                    # 计算逐元素 MSE: (pred - label)^2
                    # mse_loss = (outputs - labels) ** 2  # [batch_size, 201]
                    
                    # # 计算权重: weight = exp(3 * label)
                    # # 注意：labels 是 [0,1] 范围内的值（因为你的输出用了 Sigmoid）
                    # weights = torch.exp(4.0 * labels)  # [batch_size, 201]
                    
                    # # 加权损失
                    # weighted_loss = weights * mse_loss  # [batch_size, 201]
                    
                    # # 对整个 batch 和序列求平均（你也可以只对序列平均，再 batch 平均）
                    # loss = weighted_loss.mean()

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
                    # 计算并记录验证集的MAE和RMSE（对整个序列计算）
                    mae = np.mean(np.abs(preds - labels))
                    rmse = np.sqrt(np.mean((preds - labels) ** 2))
                    
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)
                    
                    # 保存最佳模型（只保存一个基于Loss的模型）
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print(f"New best model: {best_loss:.6f}")
                        torch.save(model.state_dict(), 'best_model_trans_full_m_now.pth')
            
    print(f'Best Val Loss: {best_loss:.4f}')
    return model

# 打印样本数据函数（修改以适应序列输出）
def print_sample_data(patterns, responses, normalizer, num_samples=5):
    print("\n" + "="*60)
    print("SAMPLE DATA ANALYSIS")
    print("="*60)
    
    # 打印原始数据
    print(f"原始参数范围检查 (前{num_samples}个样本):")
    for i in range(min(num_samples, len(patterns))):
        original_params = patterns[i]
        sequence_values = responses[i]  # 现在是201个值的数组
        print(f"样本 {i+1}:")
        print(f"  原始参数: {original_params}")
        print(f"  序列长度: {len(sequence_values)}")
        print(f"  序列值范围: [{np.min(sequence_values):.6f}, {np.max(sequence_values):.6f}]")
        
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
        print(f"  序列长度: {len(responses[i])}")
        print(f"  序列值范围: [{np.min(responses[i]):.6f}, {np.max(responses[i]):.6f}]")
    
    # 统计信息
    if len(responses) > 0:
        flat_responses = responses.flatten()
        print(f"\n数据集统计信息:")
        print(f"  总样本数: {len(patterns)}")
        print(f"  序列长度: {responses.shape[1]}")
        print(f"  全局值范围: [{np.min(flat_responses):.6f}, {np.max(flat_responses):.6f}]")
        print(f"  全局均值: {np.mean(flat_responses):.6f}")
        print(f"  全局标准差: {np.std(flat_responses):.6f}")
    print("="*60 + "\n")

# 主程序
if __name__ == '__main__':
    # 超参数
    batch_size = 512
    num_epochs = 120
    learning_rate = 1e-3
    # 创建TensorBoard日志目录
    log_dir = f"runs/transformer_experiment_m_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    # 创建归一化器
    normalizer = ParameterSpecificNormalizer(param_bounds)
    
    # 加载数据 - 使用新的完整序列加载函数
    print("Loading main data...")
    pattern_main, spectrum_main = load_cd_data("C:/data/pra", max_samples=10000)
    print(f"Loaded main {len(pattern_main)} samples with sequence length {spectrum_main.shape[1]}")

    # 打印样本数据
    # print_sample_data(pattern_main, spectrum_main, normalizer)

    # 直接使用8-2划分
    x_train, x_val, y_train, y_val = train_test_split(pattern_main, spectrum_main, test_size=0.2, random_state=42)

    # 打印划分后的信息
    print(f"\n数据集划分信息:")
    print(f"  训练集样本数: {len(x_train)}")
    print(f"  验证集样本数: {len(x_val)}")
    print(f"  序列长度: {y_train.shape[1]}")

    # 创建数据集和数据加载器
    image_datasets = {
        'train': EPDataset(x_train, y_train, normalizer),
        'val': EPDataset(x_val, y_val, normalizer)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # 模型初始化 - 修改为输出201个值
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EdgeLengthTransformerRegressor(
        seq_len=10,
        embedding_dim=64,
        d_model=128, 
        nhead=2, 
        num_layers=4,
        dropout=0.1,
        num_outputs=201  # 输出201个m值
    ).to(device)    

    # 损失函数和优化器
    criterion = nn.MSELoss(reduction='none')  # 支持序列输出的损失函数
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
    print(f"Output dimensions: {model.num_outputs} (full sequence of m values)")
    print("="*60 + "\n")

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, writer)
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看训练过程:")
    print(f"tensorboard --logdir=runs")

    import matplotlib.pyplot as plt  # 新增
    # ==============================
    # 📊 训练后：绘制预测 vs 真实值
    # ==============================
    print("\n" + "="*60)
    print("GENERATING PREDICTION PLOTS")
    print("="*60)

    # 创建保存目录
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    model.eval()

    # 50个样本
    num_examples = 50
    val_dataset = image_datasets['val']
    indices = np.random.choice(len(val_dataset), size=num_examples, replace=False)

    for i, idx in enumerate(indices):
        # 获取原始（归一化）输入和真实输出
        struct_norm, spectrum_true = val_dataset[idx]  # struct_norm: [10], spectrum_true: [201]
        struct_norm = struct_norm.unsqueeze(0).to(device)  # [1, 10]

        # 自回归预测（不使用 teacher forcing）
        with torch.no_grad():
            spectrum_pred = model(struct_norm)  # [1, 201]
        
        spectrum_pred = spectrum_pred.squeeze(0).cpu().numpy()  # [201]
        spectrum_true = spectrum_true.numpy()  # [201]

        # 绘图
        plt.figure(figsize=(12, 5))
        x_axis = np.arange(201)
        plt.plot(x_axis, spectrum_true, 'b-', linewidth=2, label='Ground Truth')
        plt.plot(x_axis, spectrum_pred, 'r--', linewidth=2, label='Prediction')
        plt.xlabel('Sequence Index')
        plt.ylabel('Response Value')
        plt.title(f'Example {i+1}: Input Parameters = {normalizer.denormalize(struct_norm.cpu().numpy()[0])}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存
        plot_path = os.path.join(plot_dir, f'prediction_example_{i+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"\nAll plots saved to '{plot_dir}' directory.")
    print("Done!")

    