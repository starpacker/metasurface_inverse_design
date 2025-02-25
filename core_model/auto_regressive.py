import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import time
from MetasurfaceCNN import Meta_CNN_Net,Meta_CRNNAG_Net
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
import random
import torch.nn.functional as F
from ipdb import set_trace

class new_pattern:
    def __init__(self, pra):
        # 参数初始化
        self.pra = pra
        self.pra1 = pra[0]
        self.pra2 = pra[1]
        self.pra3 = pra[2]
        self.pra4 = pra[3]
        self.pra5 = pra[4]
        self.pra6 = pra[5]
        self.pra7 = pra[6]
        self.pra8 = pra[7]

    # done
    def get_pattern(self):
        pattern = np.zeros((400, 400), dtype=int)
        for xi in range(pattern.shape[0]):
            for yj in range(pattern.shape[1]):

                if xi >= self.pra1 and xi <= self.pra1 + self.pra2:
                    if yj >= self.pra4 + self.pra7 + self.pra8 and \
                            yj <= self.pra4 + self.pra7 + self.pra8 + self.pra3:
                        pattern[xi, yj] = 1

                if xi >= self.pra5 and xi <= self.pra5 + self.pra6:
                    if yj >= self.pra7 and yj <= self.pra7 + self.pra8:
                        pattern[xi, yj] = 1

        return np.transpose(pattern)
    
def get_para_random_pra_test(unit=400):
    
    max_unit = unit - 10
    min_unit = 20

    pra5 = random.randint(min_unit, max_unit - 40)
    pra7 = random.randint(min_unit, max_unit - 200)

    pra6 = random.randint(min_unit, max_unit - pra5)
    pra8 = random.randint(min_unit, max_unit - pra7 - 180)

    pra4 = random.randint(min_unit, max_unit - pra7 -pra8 - 80)

    pra1 = random.randint(min_unit, max_unit - 40)
    pra2 = random.randint(min_unit, max_unit - pra1)
    pra3 = random.randint(min_unit, max_unit - pra4 - pra7 -pra8)

    return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8]

class PatternRects:
    # 参数初始化
    def __init__(self, pra):
        # 参数初始化
        self.pra = pra
        self.pra1 = pra[0]
        self.pra2 = pra[1]
        self.pra3 = pra[2]
        self.pra4 = pra[3]
        self.pra5 = pra[4]
        self.pra6 = pra[5]
        self.pra7 = pra[6]
        self.pra8 = pra[7]
        self.pra9 = pra[8]
        self.pra10 = pra[9]

    # done
    def get_pattern(self):
        pattern = np.zeros((400, 400), dtype=int)
        for xi in range(pattern.shape[0]):
            for yj in range(pattern.shape[1]):
                if xi >= self.pra1 and xi <= self.pra1 + self.pra3:
                    if yj >= self.pra10 + self.pra9 + self.pra6 + self.pra5 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5 + self.pra2:
                        pattern[xi, yj] = 1
                if xi >= self.pra1 and xi <= self.pra1 + self.pra4:
                    if yj >= self.pra10 + self.pra9 + self.pra6 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5:
                        pattern[xi, yj] = 1
                if xi >= self.pra7 and xi <= self.pra7 + self.pra8:
                    if yj >= self.pra10 and yj <= self.pra10 + self.pra9:
                        pattern[xi, yj] = 1
        return torch.from_numpy(np.transpose(pattern))
    
def get_random_pra(unit=400):
    # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
    pra1 = random.randrange(30, 160, 1)
    pra3 = random.randrange(40, 80, 1)
    pra4 = random.randrange(80, unit - 30 - pra1, 1)

    pra7 = random.randrange(30, 150, 1)
    pra8 = random.randrange(40, unit - 30 - pra7, 1)

    index = True

    while index:
        pra10 = random.randrange(30, 100, 1)
        pra9 = random.randrange(40, 100, 1)
        pra6 = random.randrange(40, 100, 1)
        pra5 = random.randrange(40, 100, 1)
        pra2 = random.randrange(40, 100, 1)
        if pra10 + pra9 + pra6 + pra5 + pra2 <= unit:
            return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
        else:
            index = True

class SSIMLoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, pred, target):
        # Binarize the predicted output for binary segmentation
        pred = (pred.softmax(dim=1)[:, 1:2] > 0.5).float()

        target = target.unsqueeze(1)

        # Check if shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predictions and targets must have the same shape, but got {pred.shape} and {target.shape}")
        return 1 - self.ssim(pred, target.float())

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

class ResidualUpBlock(nn.Module):
    """带残差连接的上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        return F.relu(x + residual)
    
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
    
class HybridDecoder(nn.Module):
    def __init__(self, latent_dim=201*6, init_channels=512):  # 400x400 output
        super().__init__()
        self.latent_dim = latent_dim
        self.init_channels = init_channels
        
        # 全连接层将特征向量转换为卷积需要的空间维度
        # from 1206 to 512 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, init_channels * 4 * 4),
            nn.BatchNorm1d(init_channels * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 上采样模块
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(init_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            
            # 8x8 -> 16x16
            ResidualUpBlock(256, 128),

            # self attention 
            SelfAttention(128),
            
            # 16x16 -> 32x32
            ResidualUpBlock(128, 64),

            # # new added
            # SelfAttention(64),

            
            # 32x32 -> 64x64
            ResidualUpBlock(64, 32),
            
            # 64x64 -> 128x128
            ResidualUpBlock(32, 16),
            
            # 128x128 -> 256x256
            ResidualUpBlock(16, 8),
            
            # 256x256 -> 400x400 (使用插值调整非2^n尺寸)
            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False),
            nn.Conv2d(8, 2, kernel_size=3, padding=1)  # 输出2通道用于二值选择
        )  
        
        self.tau = 1.0  # Gumbel-Softmax温度参数
        self.training = True

    def forward(self, x):
        # 输入形状: (batch, 6, 201)
        batch_size = x.size(0)
        
        # 展平特征
        # x = x.view(batch_size, -1)  # (batch, 6*201)

        x = x.reshape(batch_size, -1)  # especially for auto_regressive.py
        
        # 全连接层
        x = self.fc(x)
        x = x.view(batch_size, self.init_channels, 4, 4)  # (batch, 512, 4,4)
        
        # 上采样过程
        logits = self.deconv_layers(x)  # (batch, 2, 400,400)

        # if self.training:
        #     return F.gumbel_softmax(logits, tau=self.tau, hard=False)
        # else:
        #     return logits

        return logits
    
class EncoderDecoderTrainer:
    def __init__(self, device, learning_rate, num_epochs, model_path, final_model_path):
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = 32
        self.batch_num = 5
        self.model_path = model_path
        self.final_model_path = final_model_path

        # Initialize forward and backward models
        self.forward_model = Meta_CRNNAG_Net().to(self.device)
        self.forward_model.CNNLayer[0].layer3[0].relu3 = nn.ELU()
        checkpoint = torch.load('/data/group_003/yjh/logs/b512_adam_lr0.001_c1281WCRNNAG200epoch/checkpoint_max.pth', map_location="cpu")
        self.forward_model.load_state_dict(checkpoint['net'])
        self.forward_model.eval()

        self.back_model = HybridDecoder().to(self.device)

        # Training components
        self.optimizer = AdamW(self.back_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = OneCycleLR(self.optimizer, 
                                   max_lr=self.learning_rate,
                                   steps_per_epoch=self.batch_num,
                                   epochs=self.num_epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=10, delta=0.01)
        self.ssimloss = SSIMLoss(device)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # Other parameters
        self.best_val_loss = np.inf 

    def pixel_loss(self,predicted, patterns):
        # MSE loss for pixel-wise difference
        mse_loss = torch.nn.MSELoss()
        return mse_loss(predicted, patterns)
    
    def boundary_pattern(self,predicted, patterns):
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

    def manul_loss(self,probs,patterns):
        #  design to optimize the loss function

        predicted = probs[:,1,:,:].float()
        side_loss = self.boundary_pattern(predicted,patterns.float())
        direct_loss = self.pixel_loss(predicted,patterns.float())

        return direct_loss, side_loss
    
    def compute_loss(self,logits, probs, patterns, epoch):

        ce_loss = self.criterion(logits,patterns)

        ssim_loss = self.ssimloss(logits,patterns)
        ssim_weight = 0.2

        direct_loss, side_loss= self.manul_loss(probs,patterns)
        direct_weight = 5
        side_weight = 1 

        loss = (
            ce_loss + 
            direct_loss * direct_weight + 
            ssim_loss * ssim_weight + 
            side_loss * side_weight
        )
        return loss
    
    def compute_loss_val(self,logits, probs, patterns, epoch):

        ce_loss = self.criterion(logits,patterns)
        ssim_loss = self.ssimloss(logits,patterns)
        ssim_weight = 1

        direct_loss, side_loss= self.manul_loss(probs,patterns)
        direct_weight = 10
        side_weight = 1

        if epoch % 4 == 0:
            print('epoch',epoch)
            print('ce_loss:',ce_loss)                 
            print('ssim_loss',ssim_loss * ssim_weight) 
            print('direct_loss',direct_loss * direct_weight)
            print('side_loss',side_loss * side_weight)

        loss = (
            ce_loss + 
            direct_loss * direct_weight + 
            ssim_loss * ssim_weight + 
            side_loss * side_weight
        )
        return {
            "total_loss": loss,
            "ce_loss": ce_loss,
            "ssim_loss": ssim_loss,
            "pixel_loss": direct_loss,
            "boundary_loss": side_loss,
        }
    
    def forward_step(self,num):  
        patterns = []
        spectra = []
        
        for _ in range(num):
            # raw_data = get_para_random_pra_test()
            raw_data = get_random_pra()
            pattern = PatternRects(raw_data).get_pattern()
            patterns.append(pattern)

        patterns = torch.stack(patterns).to(self.device)  # [B, 400, 400]
        trg = torch.rand(num, 6, 201).to(self.device)  # [B, 6, 201]
        with torch.no_grad():
            spectrum = self.forward_model(patterns.float().unsqueeze(1),trg)
        return patterns, spectrum

    def backward_step(self,pattern,spectrum):

        spectra = spectrum.float().to(self.device)
        patterns = pattern.long().to(self.device)

        with torch.cuda.amp.autocast():
            logits = self.back_model(spectra)  # shape (B, 6, 201)

            # 确保 outputs 是 float 类型
            logits = logits.float()
            # 确保 patterns 是 long 类型
            patterns = patterns.long()

            probs = F.softmax(logits, dim=1)


        return logits,probs
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.back_model.train()
            self.back_model.training = True
            train_loss = 0.0
            for j in range(self.batch_num):
                # 一个epoch存在一个batch
                # train
                
                self.optimizer.zero_grad()
                pattern, spectrum = self.forward_step(num=self.batch_size)

                logits, probs = self.backward_step(pattern,spectrum)

                loss = self.compute_loss(logits, probs, pattern, epoch)
                self.scaler.scale(loss).backward()
                nn_utils.clip_grad_norm_(self.back_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()
            train_loss += loss.item() 

            # eval
            self.back_model.eval()
            self.back_model.training = False
            val_loss = 0.0
            val_ce_loss = 0.0
            val_ssim_loss = 0.0
            val_boundary_loss = 0.0
            val_pixel_loss = 0.0

            with torch.no_grad():
                pattern, spectrum = self.forward_step(num = 3)

                spectra = spectrum.float().to(device)
                patterns = pattern.squeeze(1).long().to(device)

                logits = self.back_model(spectra)

                # 确保 logits 是 float 类型
                logits = logits.float()

                # 确保 patterns 是 long 类型
                patterns = patterns.long()
                probs = F.softmax(logits, dim=1)

                loss_dict = self.compute_loss_val(logits, probs, patterns, epoch)

                val_loss += loss_dict["total_loss"].item() 
                val_ssim_loss += loss_dict["ssim_loss"].item() 
                val_ce_loss += loss_dict["ce_loss"].item() 
                val_pixel_loss += loss_dict["pixel_loss"].item() 
                val_boundary_loss += loss_dict["boundary_loss"].item() 
            
            # 打印训练信息
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}")
            
            if epoch >= num_epochs //2:
                self.early_stopping(val_loss)

            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break
            

# Example usage
if __name__ == "__main__":
    gpu_num = 0
    # Initialize parameters and data loaders
    device = torch.device(f"cuda:{gpu_num}")
    learning_rate = 1e-4
    num_epochs = 100
    model_path = "best_model.pth"
    final_model_path = "final_model.pth"
    
    trainer = EncoderDecoderTrainer(device, learning_rate, num_epochs, model_path, final_model_path)
    trainer.train()
