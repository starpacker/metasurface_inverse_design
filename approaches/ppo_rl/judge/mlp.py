import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import math
import gc
import random
import matplotlib.pyplot as plt

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
        params = np.array(params)
        original_shape = params.shape
        if len(original_shape) == 1:
            params = params.reshape(1, -1)
        
        normalized = np.zeros_like(params, dtype=np.float32)
        for i in range(self.num_params):
            min_val, max_val = self.param_bounds[i]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
            normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        
        if len(original_shape) == 1:
            return normalized[0]
        return normalized
    
    def denormalize(self, normalized_params):
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

# ================================
# ✅ MLP 模型替代 Transformer
# ================================
class EdgeLengthMLPRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[512, 1024, 512, 256], output_dim=201, dropout=0.2):
        super(EdgeLengthMLPRegressor, self).__init__()
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 确保输出在 [0,1]
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class EPDataset(Dataset):
    def __init__(self, struct, spectrum, normalizer):
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

def load_cd_data(data_path='C:/data/pra', max_samples=10000):
    pra_list = []
    cd_list = []

    file_paths = list(glob.iglob(os.path.join(data_path, "*.txt")))
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

        if op[-1, 1] > 750:
            continue

        r_lr_real = op[:, 10]
        r_lr_imag = op[:, 11]
        r_rl_real = op[:, 12]
        r_rl_imag = op[:, 13]
        r_rr_real = op[:, 14]
        r_rr_imag = op[:, 15]

        r_lr = r_lr_real ** 2 + r_lr_imag ** 2
        r_rl = r_rl_real ** 2 + r_rl_imag ** 2
        r_rr = r_rr_real ** 2 + r_rr_imag ** 2

        m_response = np.abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr + 1e-8)  # 防止除零

        if len(m_response) != 201:
            print(op[-1,1], len(m_response))
            continue
        pra_list.append(pra)
        cd_list.append(m_response)
        count += 1

    del file_paths
    gc.collect()
    print(f'Loaded {len(pra_list)} samples with sequence length {len(cd_list[0])}')
    return np.array(pra_list), np.array(cd_list)

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, writer=None):
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            counter = 0
            for inputs, labels in dataloaders[phase]:
                counter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    mse_loss = (outputs - labels) ** 2
                    weights = torch.exp(4.0 * labels)
                    weighted_loss = weights * mse_loss
                    loss = weighted_loss.mean()

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if phase == 'val':
                    all_preds.append(outputs.cpu().detach())
                    all_labels.append(labels.cpu().detach())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.6f}')
            
            if writer:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                if phase == 'val':
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                    
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                    preds = torch.cat(all_preds).numpy()
                    labels = torch.cat(all_labels).numpy()
                    mae = np.mean(np.abs(preds - labels))
                    rmse = np.sqrt(np.mean((preds - labels) ** 2))
                    
                    writer.add_scalar('Metrics/MAE', mae, epoch)
                    writer.add_scalar('Metrics/RMSE', rmse, epoch)
                    
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print(f"New best model: {best_loss:.6f}")
                        torch.save(model.state_dict(), 'best_model_mlp_full_sequence.pth')
            
    print(f'Best Val Loss: {best_loss:.6f}')
    return model

def print_sample_data(patterns, responses, normalizer, num_samples=5):
    print("\n" + "="*60)
    print("SAMPLE DATA ANALYSIS")
    print("="*60)
    
    for i in range(min(num_samples, len(patterns))):
        original_params = patterns[i]
        sequence_values = responses[i]
        print(f"样本 {i+1}:")
        print(f"  原始参数: {original_params}")
        print(f"  序列长度: {len(sequence_values)}")
        print(f"  序列值范围: [{np.min(sequence_values):.6f}, {np.max(sequence_values):.6f}]")
        
        for j, param_val in enumerate(original_params):
            min_val, max_val = param_bounds[j]
            if not (min_val <= param_val <= max_val):
                print(f"  ⚠️  参数{j}超出范围: {param_val} (应为 {min_val}-{max_val})")
    
    print(f"\n归一化后数据 (前{num_samples}个样本):")
    normalized_patterns = normalizer.normalize(patterns[:num_samples])
    for i in range(len(normalized_patterns)):
        print(f"样本 {i+1}:")
        print(f"  归一化参数: {normalized_patterns[i]}")
        print(f"  序列长度: {len(responses[i])}")
        print(f"  序列值范围: [{np.min(responses[i]):.6f}, {np.max(responses[i]):.6f}]")
    
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
    batch_size = 256
    num_epochs = 120
    learning_rate = 1e-3
    log_dir = f"runs/mlp_experiment_full_sequence_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    normalizer = ParameterSpecificNormalizer(param_bounds)
    
    print("Loading main data...")
    pattern_main, spectrum_main = load_cd_data("C:/data/pra", max_samples=6000)
    print(f"Loaded main {len(pattern_main)} samples with sequence length {spectrum_main.shape[1]}")

    x_train, x_val, y_train, y_val = train_test_split(pattern_main, spectrum_main, test_size=0.2, random_state=42)

    print(f"\n数据集划分信息:")
    print(f"  训练集样本数: {len(x_train)}")
    print(f"  验证集样本数: {len(x_val)}")
    print(f"  序列长度: {y_train.shape[1]}")

    image_datasets = {
        'train': EPDataset(x_train, y_train, normalizer),
        'val': EPDataset(x_val, y_val, normalizer)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EdgeLengthMLPRegressor(
        input_dim=10,
        hidden_dims=[512, 1024, 512, 256],
        output_dim=201,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Output dimensions: {model.output_dim} (full sequence of m values)")
    print("="*60 + "\n")

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, writer)
    
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看训练过程:")
    print(f"tensorboard --logdir=runs")

    # ==============================
    # 📊 训练后：绘制预测 vs 真实值
    # ==============================
    print("\n" + "="*60)
    print("GENERATING PREDICTION PLOTS")
    print("="*60)

    plot_dir = "plotss"
    os.makedirs(plot_dir, exist_ok=True)

    best_model_path = 'best_model_mlp_full_sequence.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model state.")

    model.eval()
    val_dataset = image_datasets['val']
    num_examples = 20
    indices = np.random.choice(len(val_dataset), size=num_examples, replace=False)

    for i, idx in enumerate(indices):
        struct_norm, spectrum_true = val_dataset[idx]
        struct_norm = struct_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            spectrum_pred = model(struct_norm)
        
        spectrum_pred = spectrum_pred.squeeze(0).cpu().numpy()
        spectrum_true = spectrum_true.numpy()

        plt.figure(figsize=(12, 5))
        x_axis = np.arange(201)
        plt.plot(x_axis, spectrum_true, 'b-', linewidth=2, label='Ground Truth')
        plt.plot(x_axis, spectrum_pred, 'r--', linewidth=2, label='Prediction')
        plt.xlabel('Sequence Index')
        plt.ylabel('Response Value')
        denorm_params = normalizer.denormalize(struct_norm.cpu().numpy()[0])
        plt.title(f'Example {i+1}: Input = {np.round(denorm_params, 1)}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f'prediction_example_{i+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"\nAll plots saved to '{plot_dir}' directory.")
    print("Done!")