# 这段代码只是从spectrum to parameter
import torch
import torch.nn as nn
import numpy as np
import wandb
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import sys
import time
from ipdb import set_trace

def get_now_time():
    """获取当前时间（以可读格式返回）"""
    return time.strftime("%Y_%m_%d_%H_%M", time.localtime())

# size of pattern : 400 * 400
# size of spectrum after process : 6 * 201
# size of batch : 64
batch_size_global = 32

class EPDataset(Dataset):
    def __init__(self,spectrum, pattern):
        # 这里的张量必须有相同的第一维度长度
        # shape of spectrum : [B,6,201]
        # shape of pattern : [B,400,400] 
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)
        self.pattern = torch.tensor(pattern, dtype=torch.long)

    def __getitem__(self, idx):
        spec = self.spectrum[idx]
        # 不能添加噪声！！！ 数据很小，并且比较敏感

        return spec, self.pattern[idx]

    def __len__(self):
        return self.spectrum.shape[0]
    
class EP_normalized_Dataset(Dataset):
    def __init__(self, spectrum, pattern, value_list):
        # 归一化处理
        self.spectrum = torch.tensor(spectrum, dtype=torch.float32)
        self.pattern = torch.tensor(pattern, dtype=torch.long)
    
        B, Channel, L = self.spectrum.shape  # B: batch, C: channels (6), L: length (201)
        
        # 逐通道归一化
        for c in range(Channel):
            min_val, max_val = value_list[c]
            # min_val = current_channel.min(dim=1, keepdim=True)[0]  # 每个样本在该通道的最小值
            # max_val = current_channel.max(dim=1, keepdim=True)[0]  # 每个样本在该通道的最大值
            self.spectrum[:, c, :] = (self.spectrum[:, c, :] - min_val) / (max_val - min_val)
    
    def __getitem__(self, idx):
        return self.spectrum[idx], self.pattern[idx]
    
    def __len__(self):
        return self.spectrum.shape[0]

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # 允许的验证损失不下降的次数
        self.delta = delta  # 验证损失下降的最小阈值
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Attention_with_loc(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        
        # 将通道数降低以降低计算量
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数
        
        # LayerNorm用于稳定训练过程
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])  # 在最后输出维度进行规范化
        
        # 位置编码（采用简单的二维位置编码）
        self.pos_encoding = self.create_positional_encoding(in_channels)
    
    def create_positional_encoding(self, in_channels):
        """
        生成位置编码
        """
        height = 400  # 假设图像高度
        width = 400   # 假设图像宽度
        pe = torch.zeros(height, width, in_channels)
        
        for i in range(height):
            for j in range(width):
                for k in range(in_channels):
                    pe[i, j, k] = torch.sin(i / (10000 ** (k / in_channels))) if k % 2 == 0 else torch.cos(j / (10000 ** ((k - 1) / in_channels)))
        
        return pe.unsqueeze(0)  # Batch维度
    
    def forward(self, x):
        """
        x: shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # 计算查询、键、值
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        proj_key   = self.key_conv(x).view(B, -1, H * W)                       # (B, C//8, H*W)
        energy = torch.bmm(proj_query, proj_key)                               # (B, H*W, H*W)
        
        # 加上位置编码（相对位置编码）
        energy = energy + self.pos_encoding[:, :, :H * W].view(1, -1, H * W)  # 添加位置信息
        
        # Softmax计算注意力权重
        attention = F.softmax(energy, dim=-1)                                  # (B, H*W, H*W)
        
        # 计算加权值
        proj_value = self.value_conv(x).view(B, C, -1)                         # (B, C, H*W)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, H*W)
        out = out.view(B, C, H, W)
        
        # 残差连接与LayerNorm
        out = self.layer_norm(out + x)  # 残差连接并进行LayerNorm
        
        out = self.gamma * out  # 缩放

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 将通道数降低以降低计算量
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数

    def forward(self, x):
        """
        x: shape (B, C, H, W)
        """
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        proj_key   = self.key_conv(x).view(B, -1, H * W)                       # (B, C//8, H*W)
        energy = torch.bmm(proj_query, proj_key)                               # (B, H*W, H*W)
        attention = F.softmax(energy, dim=-1)                                  # (B, H*W, H*W)
        proj_value = self.value_conv(x).view(B, C, -1)                           # (B, C, H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, H*W)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x  # 残差连接
        return out
    
class ResNetUpBlock(nn.Module):
    """基于ResNet的上采样残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 残差路径（通道数或尺寸变化时需调整）
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))
    
class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        return self.upsample(x)
    
class OptimizedDecoder(nn.Module):
    def __init__(self, latent_dim=201*6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512,4,4)),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(

            ResNetUpBlock(512,256),        # 4x4 ->8x8

            SelfAttention(256),

            ResNetUpBlock(256,128),        # 8x8->16x16

            PixelShuffleUp(128,64),        # 16x16->32x32

            ResNetUpBlock(64,32),          # 保持尺寸

            SelfAttention(32),

            nn.Upsample(scale_factor=2, mode='bilinear'),  # 32x32->64x64

            nn.Conv2d(32,16,3,padding=1),

            ResNetUpBlock(16,8),

            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False),
            
            nn.Conv2d(8, 2, kernel_size=3, padding=1)  # 输出2通道用于二值选择
        )

    def forward(self, x):
        x = x.view(batch_size_global, -1)  # Flatten the last two dimensions
        x = self.fc(x)
        return self.decoder(x)
    

# pattern = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_data_matrix2000.pt')
# spectrum = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_spectrum2000.pt')
pattern = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_data_matrix2000.pt')
spectrum = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_spectrum2000.pt')

# 使用抽样索引创建子集
x_train, x_val, y_train, y_val = train_test_split(
    spectrum[:, 1:7, :],  # 确保 spectrum_test 是张量
    pattern,             # 确保 pattern_test 是张量
    test_size=0.1,
    random_state=42  # 确保结果可复现
)

value_list = [(-0.80770737, 0.824261), (-0.79511195, 0.85141766), (-0.8091829, 0.75830907), (-0.85016185, 0.8366283), (-0.7531471, 0.97804517), (-0.71447146, 0.8164543)]

train_set = EP_normalized_Dataset(
    spectrum=x_train,
    pattern=y_train,
    value_list=value_list
    )

test_set = EP_normalized_Dataset(
    spectrum=x_val,
    pattern=y_val,
    value_list=value_list
    )

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size_global,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size_global,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=True
)



def train_model():
    print("time for training ~~")
    now_time = get_now_time()
    model_path = f"D:/subject/physics/AI4S/project1/model/best_designer_model_{now_time}.pth"
    final_model_path = f'D:/subject/physics/AI4S/project1/model/final_designer_model_{now_time}.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    num_epochs = 40
    learning_rate = 0.001
    best_val_loss = float('inf')

    # 初始化
    designer_model = OptimizedDecoder().to(device)
    optimizer = AdamW(designer_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                          steps_per_epoch=len(train_data_loader), 
                          epochs=num_epochs)
    criterion = nn.CrossEntropyLoss()  # 用于像素级分类
    early_stopping = EarlyStopping(patience=6, delta=0.01)

    
    def pixel_loss(predicted, patterns):
        # MSE loss for pixel-wise difference
        mse_loss = torch.nn.MSELoss()
        return mse_loss(predicted, patterns)
    
    def boundary_pattern(predicted, patterns):
        # 计算预测和真实图像的梯度
        dy_pred, dx_pred = torch.gradient(predicted, dim=[1, 2])  # 垂直和水平梯度
        dy_real, dx_real = torch.gradient(patterns, dim=[1, 2])
        
        edge_threshold = 0.5  # 根据实际数据调整阈值
        edge_mask = (dx_real.abs() >= edge_threshold) | (dy_real.abs() >= edge_threshold)
        
        # 计算边缘位置的梯度差异
        loss_dx = torch.abs(dx_pred[edge_mask] - dx_real[edge_mask]).mean()
        loss_dy = torch.abs(dy_pred[edge_mask] - dy_real[edge_mask]).mean()
        boundary_loss = loss_dx + loss_dy
        
        return boundary_loss

    def manul_loss(probs,patterns):
        #  design to optimize the loss function

        predicted = probs[:,1,:,:].float()
        side_loss = boundary_pattern(predicted,patterns.float())
        direct_loss = pixel_loss(predicted,patterns.float())

        return direct_loss, side_loss

    def compute_loss(logits, probs, patterns, epoch):

        ce_loss = criterion(logits,patterns)

        # tv_loss = total_variation_loss(probs)
        # tv_weight = max(2 * (1 - epoch/num_epochs), 1)  # 随训练逐渐降低


        direct_loss, side_loss= manul_loss(probs,patterns)
        direct_weight = 5
        side_weight = 1 



        # loss = ce_loss + tv_weight * tv_loss + ssim_loss * ssim_weight + direct_loss * direct_weight + side_loss * side_weight
        # loss = ce_loss + tv_weight * tv_loss

        loss = (
            ce_loss + 
            direct_loss * direct_weight + 
            side_loss * side_weight
        )
        return loss
    

    def compute_loss_test(logits, probs, patterns, epoch):

        direct_loss, side_loss= manul_loss(probs,patterns)
        direct_weight = 5 - epoch * 0.2
        side_weight = 1 + epoch * 0.06

        return direct_loss * direct_weight + side_loss * side_weight
    

    
    def compute_loss_val(logits, probs, patterns, epoch):

        ce_loss = criterion(logits,patterns)

        direct_loss, side_loss= manul_loss(probs,patterns)
        direct_weight = 10
        side_weight = 1

        if epoch % 4 == 0:
            print('epoch',epoch)
            print('ce_loss:',ce_loss)           
            print('direct_loss',direct_loss * direct_weight)
            print('side_loss',side_loss * side_weight)


        # loss = ce_loss + tv_weight * tv_loss + ssim_loss * ssim_weight + direct_loss * direct_weight + side_loss * side_weight
        # loss = ce_loss + tv_weight * tv_loss

        loss = (
            ce_loss + 
            direct_loss * direct_weight +  
            side_loss * side_weight
        )
        return {
            "total_loss": loss,
            "ce_loss": ce_loss,
            "pixel_loss": direct_loss,
            "boundary_loss": side_loss,
        }

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    wandb.init(project="metasurface-design")
    wandb.watch(designer_model)

    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        # 训练阶段
        designer_model.train()
        designer_model.training = True
        train_loss = 0.0
        for batch_idx, (spectra, patterns) in enumerate(train_data_loader):
            spectra = spectra.float().to(device)        # (B,6,201)
            patterns = patterns.squeeze(1).long().to(device)  # (B,400,400) 移除通道维度

            optimizer.zero_grad()

            # 混合精度训练
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = designer_model(spectra)  # shape (B,6,201)

                    # 确保 outputs 是 float 类型
                    logits = logits.float()
                    # 确保 patterns 是 long 类型
                    patterns = patterns.long()

                    probs = F.softmax(logits, dim=1)

                    # loss = compute_loss(logits, probs, patterns, epoch)
                    loss = compute_loss(logits, probs, patterns, epoch)

                scaler.scale(loss).backward()
                nn_utils.clip_grad_norm_(designer_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = designer_model(spectra)
                # 确保 outputs 是 float 类型
                logits = logits.float()

                # 确保 patterns 是 long 类型
                patterns = patterns.long()
                probs = F.softmax(logits, dim=1)

                # loss = compute_loss(logits, probs, patterns, epoch)
                loss = compute_loss(logits, probs, patterns, epoch)

                loss.backward()
                nn_utils.clip_grad_norm_(designer_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item() * spectra.size(0)
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_data_loader.dataset)
        

        # 验证阶段
        designer_model.eval()
        designer_model.training = False
        val_loss = 0.0
        val_ce_loss = 0.0
        val_boundary_loss = 0.0
        val_pixel_loss = 0.0

        with torch.no_grad():
            for spectra, patterns in test_data_loader:
                spectra = spectra.float().to(device)
                patterns = patterns.squeeze(1).long().to(device)

                logits = designer_model(spectra)

                # 确保 logits 是 float 类型
                logits = logits.float()

                # 确保 patterns 是 long 类型
                patterns = patterns.long()
                probs = F.softmax(logits, dim=1)

                loss_dict = compute_loss_val(logits, probs, patterns, epoch)

                val_loss += loss_dict["total_loss"].item() * spectra.size(0)
                val_ce_loss += loss_dict["ce_loss"].item() * spectra.size(0)
                val_pixel_loss += loss_dict["pixel_loss"].item() * spectra.size(0)
                val_boundary_loss += loss_dict["boundary_loss"].item() * spectra.size(0)
        
        val_loss = val_loss / len(test_data_loader.dataset)
        val_ce_loss = val_ce_loss / len(test_data_loader.dataset)
        val_pixel_loss = val_pixel_loss / len(test_data_loader.dataset)
        val_boundary_loss = val_boundary_loss / len(test_data_loader.dataset)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_total_loss": val_loss,
            "val_pixel_loss": val_pixel_loss,
            "val_boundary_loss": val_boundary_loss,
            "val_crossentropy_loss": val_ce_loss
        })
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        
        # 保存最佳模型
        if val_loss < best_val_loss and val_loss < 2:
            print("------------------save best model!-----------------------")
            best_val_loss = val_loss
            
            torch.save(designer_model.state_dict(), model_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算循环耗时
        print(f"{epoch + 1} epoch completed. Time taken: {elapsed_time:.4f} seconds")
        time.sleep(0.5)
    # 最终模型保存
    torch.save(designer_model.state_dict(), final_model_path)

if __name__ == "__main__":
    train_model()
    


