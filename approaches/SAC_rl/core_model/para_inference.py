# 这段代码只是从spectrum to parameter
import torch
import torch.nn as nn
import numpy as np
import wandb
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys
import matplotlib.pyplot as plt
import time
from ipdb import set_trace

def get_now_time():
    """获取当前时间（以可读格式返回）"""
    return time.strftime("%Y_%m_%d_%H_%M", time.localtime())

# size of pattern : 400 * 400
# size of spectrum after process : 6 * 201

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
    

# 上采样会带来棋盘效应，需要更加高效的采样方式： multi-scale-fusion
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x_ca = x * ca
        
        # 空间注意力
        sa_input = torch.cat([x_ca.mean(dim=1, keepdim=True), 
                            x_ca.max(dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_att(sa_input)
        return x_ca * sa
    
class MultiScaleFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主分支使用亚像素卷积
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2倍上采样
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        
        # 跳跃分支使用插值+卷积
        self.skip_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        
        # 特征融合后的处理
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            CBAM(out_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        x_main = self.main_branch(x)
        x_skip = self.skip_branch(x)
        fused = torch.cat([x_main, x_skip], dim=1)
        return self.post_conv(fused)
    
class EnhancedDecoder(nn.Module):
    def __init__(self, latent_dim=201*6, init_channels=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 更智能的特征展开
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, init_channels*4*4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(init_channels*4*4, init_channels*4*4),
            nn.LayerNorm(init_channels*4*4)
        )
        
        # 多阶段上采样架构
        self.decoder = nn.Sequential(
            # Stage 1: 4x4 -> 8x8
            MultiScaleFusionBlock(512, 256),
            
            # Stage 2: 8x8 -> 16x16
            MultiScaleFusionBlock(256, 128),

            # CBAM(128),   #减少计算量
            
            # Stage 3: 16x16 -> 32x32
            MultiScaleFusionBlock(128, 64),
            
            # Stage 4: 32x32 -> 64x64
            MultiScaleFusionBlock(64, 32),
            
            # Stage 5: 64x64 -> 128x128
            MultiScaleFusionBlock(32, 16),
            
            # Final upsampling
            nn.Upsample(scale_factor=3.125, mode='bilinear', align_corners=False),  # 128->400
            nn.Conv2d(16, 2, kernel_size=5, padding=2),
            nn.Dropout2d(0.1)
        )
        
        # 动态温度调整
        self.tau_scheduler = lambda epoch: max(0.5 * (0.9 ** epoch), 0.1)

    def forward(self, x, current_epoch=0):
        # 动态调整Gumbel温度
        self.tau = self.tau_scheduler(current_epoch)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, 512, 4, 4)
        x = self.decoder(x)
        return F.gumbel_softmax(x, tau=self.tau, hard=not self.training)


# ssim loss cost too much resources
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




def train_model(gpu_num):
    print("time for training ~~")
    now_time = get_now_time()
    model_path = "best_designer_model.pth"
    final_model_path = "final_designer_model.pth"

    device = torch.device(f"cuda:{gpu_num}")

    # 训练参数
    num_epochs = 80
    learning_rate = 0.001
    best_val_loss = float('inf')

    # 初始化
    designer_model = HybridDecoder().to(device)
    # designer_model = EnhancedDecoder().to(device)
    optimizer = AdamW(designer_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                          steps_per_epoch=len(train_data_loader), 
                          epochs=num_epochs)
    # print(len(train_data_loader))
    # print(train_data_loader.batch_size)
    criterion = nn.CrossEntropyLoss()  # 用于像素级分类
    early_stopping = EarlyStopping(patience=8, delta=0.01)
    ssimloss = SSIMLoss(device)  
    # wandb.init(project="metasurface-design")
    # wandb.watch(designer_model)

    # def boundary_pattern(predicted,patterns):
    #     # torch size of [B,400,400]
    #     mini_predicted = torch.min(predicted,dim=1)[0] + torch.min(predicted,dim=2)[0] 
    #     mini_real = torch.min(patterns,dim=1)[0] + torch.min(patterns,dim=2)[0]
    #     maxi_predicted = torch.max(predicted,dim=1)[0] + torch.max(predicted,dim=2)[0]
    #     maxi_real = torch.max(patterns,dim=1)[0] + torch.max(patterns,dim=2)[0]

    #     boundary_loss = torch.abs(mini_predicted-mini_real) + torch.abs(maxi_predicted-maxi_real)
    #     return boundary_loss.mean() / 400

    def total_variation_loss(x):
        """
        计算全变分损失，使生成的图案在空间上更平滑。
        参数:
            x: 张量，形状为 (B, C, H, W),这里我们通常对softmax后的概率图计算TV损失
        返回:
            标量的TV损失
        """
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv
    
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

        ssim_loss = ssimloss(logits,patterns)
        ssim_weight = 0.2

        direct_loss, side_loss= manul_loss(probs,patterns)
        direct_weight = 5
        side_weight = 1 



        # loss = ce_loss + tv_weight * tv_loss + ssim_loss * ssim_weight + direct_loss * direct_weight + side_loss * side_weight
        # loss = ce_loss + tv_weight * tv_loss

        loss = (
            ce_loss + 
            direct_loss * direct_weight + 
            ssim_loss * ssim_weight + 
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

        # tv_loss = total_variation_loss(probs)
        # tv_weight = max(2 * (1 - epoch/num_epochs), 1)  # 随训练逐渐降低

        ssim_loss = ssimloss(logits,patterns)
        ssim_weight = 1

        direct_loss, side_loss= manul_loss(probs,patterns)
        direct_weight = 10
        side_weight = 1

        if epoch % 4 == 0:
            print('epoch',epoch)
            print('ce_loss:',ce_loss)         
            # print('tv_loss',tv_loss)           
            print('ssim_loss',ssim_loss * ssim_weight) 
            print('direct_loss',direct_loss * direct_weight)
            print('side_loss',side_loss * side_weight)


        # loss = ce_loss + tv_weight * tv_loss + ssim_loss * ssim_weight + direct_loss * direct_weight + side_loss * side_weight
        # loss = ce_loss + tv_weight * tv_loss

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
                    logits = designer_model(spectra)  # shape (B, 6, 201)

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
        val_ssim_loss = 0.0
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
                val_ssim_loss += loss_dict["ssim_loss"].item() * spectra.size(0)
                val_ce_loss += loss_dict["ce_loss"].item() * spectra.size(0)
                val_pixel_loss += loss_dict["pixel_loss"].item() * spectra.size(0)
                val_boundary_loss += loss_dict["boundary_loss"].item() * spectra.size(0)
        
        val_loss = val_loss / len(test_data_loader.dataset)
        val_ce_loss = val_ce_loss / len(test_data_loader.dataset)
        val_ssim_loss = val_ssim_loss / len(test_data_loader.dataset)
        val_pixel_loss = val_pixel_loss / len(test_data_loader.dataset)
        val_boundary_loss = val_boundary_loss / len(test_data_loader.dataset)

        wandb.log({
            "epoch": epoch,
            "val_pixel_loss": val_pixel_loss,
            "val_boundary_loss": val_boundary_loss,
            "val_ssim_loss": val_ssim_loss,
            "val_crossentropy_loss": val_ce_loss
        })
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        early_stopping(val_loss)

        if early_stopping.early_stop and epoch >= 30:
            print("Early stopping triggered!")
            break
        
        wandb.log({
        "train_loss": train_loss,
        ""
        "val_loss": val_loss
        })
        torch.cuda.empty_cache()
        
        
        # 保存最佳模型
        if val_loss < best_val_loss and val_loss < 2:
            print("------------------save best model!-----------------------")
            best_val_loss = val_loss
            time.sleep(0.5)
            torch.save(designer_model.state_dict(), model_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算循环耗时
        print(f"{epoch + 1} epoch completed. Time taken: {elapsed_time:.4f} seconds")
        
    # 最终模型保存
    print("save final model")
    torch.save(designer_model.state_dict(), final_model_path)

def draw_pattern(pra,name:str):
    try:
        pra = pra.detach().numpy()
    except:
        pra = pra
    fig, ax = plt.subplots()
    ax.imshow(pra, origin='lower')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.savefig(name)


def test_and_visualize(model, test_data_loader, device, num_samples=5):
    """
    测试模型并可视化预测结果
    :param model: 训练好的模型
    :param test_data_loader: 测试数据的 DataLoader
    :param device: 设备（'cpu' 或 'cuda')
    :param num_samples: 可视化的样本数量
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for spectra, patterns in test_data_loader:
            spectra = spectra.float().to(device)
            patterns = patterns.squeeze(1).long().to(device)

            # 模型预测
            logits = model(spectra)
            probs = F.softmax(logits, dim=1)
            predicted = (probs[:, 1, :, :] > 0.5).float()

            # 可视化真实图案和预测图案
            for i in range(num_samples):
                print(f"Sample {i + 1}:")
                print("Real pattern:")
                draw_pattern(patterns[i].cpu().numpy(),f"real{i+1}")  # 真实图案
                print("Predicted pattern:")
                draw_pattern(predicted[i].cpu().numpy(),f"imag{i+1}")  # 预测图案

            # 只可视化指定数量的样本
            break

if __name__ == "__main__":

    # -------------------loading data-----------------------
    # gpu_num = 2
    # pattern = torch.load('/data/group_003/yjh/data_set/tensor_data_matrix2000.pt')
    # spectrum = torch.load('/data/group_003/yjh/data_set/tensor_spectrum2000.pt')
    # # pattern = torch.load('combined_pattern.pt')
    # # spectrum = torch.load('combined_spectrum.pt')

    # x_train, x_val, y_train, y_val = train_test_split(
    #     spectrum[:, 1:7, :],  # 确保 spectrum_test 是张量
    #     pattern,             # 确保 pattern_test 是张量
    #     test_size=0.1,
    #     random_state=42  # 确保结果可复现
    # )
    # print("pattern shape",pattern.shape)
    # print("spectrum shape",spectrum.shape)

    gpu_num = 0
    # pattern = torch.load("6000_real_pattern.pt")
    # spectrum = torch.load("6000_real_spectrum.pt")
    pattern = torch.load('/data/group_003/yjh/data_set_s/patterns_synthesis_6000.pt')
    # spectrum = torch.load('/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_6000.pt') 
    spectrum = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_6000.pt') 

    pt = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_5000.pt")
    # sp = torch.load("/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_5000.pt")
    sp = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_5000.pt') 
    pattern = torch.cat((pattern,pt),dim=0)
    spectrum = torch.cat((spectrum,sp),dim=0)

    pt = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_5000_2.pt")
    # sp = torch.load("/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_5000_2.pt")
    sp = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_5000_2.pt')
    pattern = torch.cat((pattern,pt),dim=0)
    spectrum = torch.cat((spectrum,sp),dim=0)

    pt = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_3000.pt")
    # sp = torch.load("/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_3000.pt")
    sp = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_3000.pt')
    pattern = torch.cat((pattern,pt),dim=0)
    spectrum = torch.cat((spectrum,sp),dim=0)

    pt = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_3000_2.pt")
    # sp = torch.load("/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_3000_2.pt")
    sp = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_3000_2.pt')
    pattern = torch.cat((pattern,pt),dim=0)
    spectrum = torch.cat((spectrum,sp),dim=0)

    # pt = torch.load("/data/group_003/yjh/data_set_s/patterns_synthesis_6000_2.pt")
    # # sp = torch.load("/data/group_003/yjh/data_set_s/spectra_CRNNAG_synthesis_6000_2.pt")
    # sp = torch.load('/data/group_003/yjh/data_set_s/spectra_CNN_synthesis_6000_2.pt')
    # pattern = torch.cat((pattern,pt),dim=0)
    # spectrum = torch.cat((spectrum,sp),dim=0)


    # spectrum = torch.load('/data/group_003/yjh/data_set_s/spectra_CRNNAG_attention_synthesis_3000.pt') 
    # pattern = torch.load('/data/group_003/yjh/data_set_l/sythesis_patterns_3000.pt')
    # spectrum = torch.load('/data/group_003/yjh/data_set_l/sythesis_spectra_3000.pt') 

    print("pattern shape",pattern.shape)
    print("spectrum shape",spectrum.shape)
    x_train, x_val, y_train, y_val = train_test_split(
        spectrum,  # 确保 spectrum_test 是张量
        pattern,             # 确保 pattern_test 是张量
        test_size=0.1,
        random_state=42  # 确保结果可复现
    )

    # gpu_num = 0
    # num_batches = 20
    # pattern = None
    # for i in range(num_batches):
    #     filename = f"/data/group_003/data_set_b/new_data_matrix_{i}.pt"
    #     batch = torch.load(filename)
    #     if pattern is None:
    #         pattern = batch
    #     else:
    #         pattern = torch.cat((pattern, batch), dim=0)
    #     print(i+1,pattern.shape)
    # print(f"Merged array shape: {pattern.shape}")
    # spectrum = torch.load(f'/data/group_003/data_set_b/new_data_spectrum_{num_batches}000.pt')
    # print(f"spectrum shape:{spectrum.shape}")

    # x_train, x_val, y_train, y_val = train_test_split(
    #     spectrum[:, 1:7, :],  # 确保 spectrum_test 是张量
    #     pattern,             # 确保 pattern_test 是张量
    #     test_size=0.1,
    #     random_state=42  # 确保结果可复现
    # )

    # ----------------------loading data------------------

    # # 使用抽样索引创建子集
    # x_train, x_val, y_train, y_val = train_test_split(
    #     spectrum,  # 确保 spectrum_test 是张量
    #     pattern,             # 确保 pattern_test 是张量
    #     test_size=0.1,
    #     random_state=42  # 确保结果可复现
    # )



    # value_list = [(-0.80770737, 0.824261), (-0.79511195, 0.85141766), (-0.8091829, 0.75830907), (-0.85016185, 0.8366283), (-0.7531471, 0.97804517), (-0.71447146, 0.8164543)]

    # train_set = EP_normalized_Dataset(
    #     spectrum=x_train,
    #     pattern=y_train,
    #     value_list=value_list
    #     )

    # test_set = EP_normalized_Dataset(
    #     spectrum=x_val,
    #     pattern=y_val,
    #     value_list=value_list
    #     )

    train_set = EPDataset(
        spectrum=x_train,
        pattern=y_train)

    test_set = EPDataset(
        spectrum=x_val,
        pattern=y_val)



    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=64,
        shuffle=True,
        drop_last=False
    )
    train_model(gpu_num=gpu_num)
    
