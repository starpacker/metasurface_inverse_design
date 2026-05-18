# 这段代码只是从spectrum to parameter
import torch
import torch.nn as nn
import numpy as np
import wandb
# from Pattern import PatternRects
import torch.nn.functional as F
# from FDTDModel import test_pattern_rects  # from para to spectrum
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils as nn_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys
import time

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

        # # 对 spectrum 按通道进行 min-max 归一化：沿最后一维计算最小值和最大值
        # spec_min = self.spectrum.amin(dim=-1, keepdim=True)  # shape: [B, 6, 1]
        # spec_max = self.spectrum.amax(dim=-1, keepdim=True)  # shape: [B, 6, 1]
        
        # # 避免除以零，加上一个很小的数 epsilon
        # epsilon = 1e-8
        # self.spectrum = (self.spectrum - spec_min) / (spec_max - spec_min + epsilon)

    # def __getitem__(self, index):
    #     return self.spectrum[index], self.pattern[index]
    def __getitem__(self, idx):
        spec = self.spectrum[idx]

        # # 随机添加噪声
        # if np.random.rand() < 0.3:
        #     spec += torch.randn_like(spec) * 0.01
        return spec, self.pattern[idx]

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
        x = x.view(batch_size, -1)  # (batch, 6*201)
        
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


    
pattern = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_data_matrix2000.pt')
spectrum = torch.load('D:/subject/physics/AI4S/project1/tensor_data_2000/tensor_spectrum2000.pt')


# # 定义抽样大小
# sample_size = 800
# # 随机抽样索引
# indices = torch.randperm(pattern.size(0))[:sample_size]

# # 使用索引直接从张量中提取子集
# spectrum = spectrum[indices]
# pattern = pattern[indices]

# 使用抽样索引创建子集
x_train, x_val, y_train, y_val = train_test_split(
    spectrum[:, 1:7, :],  # 确保 spectrum_test 是张量
    pattern,             # 确保 pattern_test 是张量
    test_size=0.1,
    random_state=42  # 确保结果可复现
)

# print(x_train[0])
# sys.exit()
# print(len(x_train))
# print(len(y_train))
# print(len(x_val))
# print(len(y_val))

# x_train, x_val, y_train, y_val = train_test_split( spectrum[:,1:7,:], pattern, test_size=0.1)


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
    drop_last=True,
    num_workers=0,
    pin_memory=True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=True
)



def train_model():
    print("time for training ~~")
    now_time = get_now_time()
    model_path = f"D:/subject/physics/AI4S/project1/model/new_best_designer_model_{now_time}.pth"
    final_model_path = f'D:/subject/physics/AI4S/project1/model/new_final_designer_model_{now_time}.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    num_epochs = 20
    learning_rate = 0.001
    best_val_loss = float('inf')

    # 初始化
    designer_model = HybridDecoder().to(device)
    # designer_model = EnhancedDecoder().to(device)
    optimizer = AdamW(designer_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                          steps_per_epoch=len(train_data_loader), 
                          epochs=num_epochs)
    criterion = nn.CrossEntropyLoss()  # 用于像素级分类
    early_stopping = EarlyStopping(patience=6, delta=0.01)
    # ssimloss = SSIMLoss(device)  


    def pixel_loss(predicted, target):
        """加权像素损失，突出矩形区域"""
        # 创建权重矩阵（矩形区域权重为3，其他区域为1）
        weights = torch.where(target > 0.5, 3.0, 1.0)
        return torch.mean(weights * (predicted - target)**2)

    def sobel_edges(img):
        """使用Sobel算子进行边缘检测"""
        # 水平/垂直方向Sobel核
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                            dtype=torch.float32, device=img.device)
        sobel_y = sobel_x.t()
        
        # 扩展维度用于卷积操作 (batch, channel, H, W)
        img = img.unsqueeze(1)
        edges_x = F.conv2d(img, sobel_x.view(1,1,3,3), padding=1)
        edges_y = F.conv2d(img, sobel_y.view(1,1,3,3), padding=1)
        return edges_x.squeeze(1), edges_y.squeeze(1)

    def boundary_loss(predicted, target):
        """改进的边缘损失函数"""
        # Sobel边缘检测
        pred_x, pred_y = sobel_edges(predicted)
        target_x, target_y = sobel_edges(target)
        
        # 动态边缘阈值（取真实图像边缘强度的中位数）
        edge_magnitude = torch.sqrt(target_x**2 + target_y**2)
        threshold = torch.median(edge_magnitude) * 0.5
        edge_mask = (edge_magnitude > threshold)
        
        # 边缘方向一致性损失
        cos_sim = F.cosine_similarity(
            torch.stack([pred_x, pred_y], dim=-1),
            torch.stack([target_x, target_y], dim=-1),
            dim=-1
        )
        direction_loss = 1 - cos_sim[edge_mask].mean()
        
        # 边缘强度匹配损失
        strength_loss = F.l1_loss(
            torch.sqrt(pred_x[edge_mask]**2 + pred_y[edge_mask]**2),
            torch.sqrt(target_x[edge_mask]**2 + target_y[edge_mask]**2)
        )
        
        return direction_loss + strength_loss

    def projection_loss(predicted, target):
        """结构投影损失（水平/垂直投影匹配）"""
        # 水平投影 (batch, H)
        hp_pred = predicted.sum(dim=2)
        hp_target = target.sum(dim=2)
        
        # 垂直投影 (batch, W)
        vp_pred = predicted.sum(dim=1)
        vp_target = target.sum(dim=1)
        
        # 形状对齐损失
        return F.mse_loss(hp_pred, hp_target) + F.mse_loss(vp_pred, vp_target)

    def relative_position_loss(predicted, target):
        """相对位置关系损失（基于质心距离）"""
        # 生成坐标网格
        B, H, W = predicted.shape
        y_coord = torch.linspace(0, 1, H, device=predicted.device).view(1, H, 1)
        x_coord = torch.linspace(0, 1, W, device=predicted.device).view(1, 1, W)
        
        # 计算质心坐标
        pred_mass = predicted.sum(dim=(1,2), keepdim=True) + 1e-8
        pred_cy = (predicted * y_coord).sum(dim=(1,2)) / pred_mass.squeeze()
        pred_cx = (predicted * x_coord).sum(dim=(1,2)) / pred_mass.squeeze()
        
        target_mass = target.sum(dim=(1,2), keepdim=True) + 1e-8
        target_cy = (target * y_coord).sum(dim=(1,2)) / target_mass.squeeze()
        target_cx = (target * x_coord).sum(dim=(1,2)) / target_mass.squeeze()
        
        # 相对位移损失
        dx_loss = F.mse_loss(pred_cx[:, None] - pred_cx, target_cx[:, None] - target_cx)
        dy_loss = F.mse_loss(pred_cy[:, None] - pred_cy, target_cy[:, None] - target_cy)
        
        return (dx_loss + dy_loss) 

    def compute_loss_val(predicted, target):
        """组合多目标损失函数"""
        pixel_loss_value = pixel_loss(predicted, target)
        boundary_loss_value = boundary_loss(predicted, target)
        # projection_loss_value = projection_loss(predicted, target)
        relative_position_loss_value = relative_position_loss(predicted, target)

        if epoch % 4 ==0:
            print("epoch",epoch)
            print("pixel_loss",pixel_loss_value * 15)
            print("boundary_loss",boundary_loss_value * 0.6)
            # print("projection_loss",projection_loss_value)
            print("relative_position_loss",relative_position_loss_value * (20 + epoch * 7))

        # 各损失项权重可调
        total_loss = (
            10 * pixel_loss_value +
            0.6 * boundary_loss_value +
            # 0.5 * projection_loss_value +
            (20 + epoch * 7) * relative_position_loss_value
        )
            # 返回单独的损失项和总损失
        return {
            "total_loss": total_loss,
            "pixel_loss": pixel_loss_value,
            "boundary_loss": boundary_loss_value,
            "relative_position_loss": relative_position_loss_value,
        }
    
    def compute_loss(predicted, target):
        """组合多目标损失函数"""
        pixel_loss_value = pixel_loss(predicted, target)
        boundary_loss_value = boundary_loss(predicted, target)
        # projection_loss_value = projection_loss(predicted, target)
        relative_position_loss_value = relative_position_loss(predicted, target)

        # 各损失项权重可调
        total_loss = (
            10 * pixel_loss_value +
            0.6 * boundary_loss_value +
            # 0.5 * projection_loss_value +
            (20 + epoch * 7) * relative_position_loss_value
        )

        # 返回单独的损失项和总损失
        return total_loss


    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    wandb.init(project="new-metasurface-design")
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
            patterns = patterns.squeeze(1).float().to(device)  # (B,400,400) 移除通道维度
            
            optimizer.zero_grad()

            # 混合精度训练
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = designer_model(spectra)  # shape (B, 2, 400,400)

                    # 确保 outputs 是 float 类型
                    logits = logits.float()

                    probs = F.softmax(logits, dim=1)
                    predicted = probs[:,1,:,:].squeeze().float()
                    
                    loss = compute_loss(predicted, patterns)

                    

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
                predicted = probs[:,1,:,:].squeeze().float()

                loss = compute_loss(predicted, patterns)
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
        val_pixel_loss = 0
        val_boundary_loss = 0
        val_projection_loss = 0
        val_relative_position_loss = 0
        with torch.no_grad():
            for spectra, patterns in test_data_loader:
                spectra = spectra.float().to(device)
                patterns = patterns.squeeze(1).float().to(device)

                logits = designer_model(spectra)

                # 确保 logits 是 float 类型
                logits = logits.float()
                if epoch % 3 == 0:
                    ce_loss = criterion(logits,patterns.long())
                    print("ce_loss",ce_loss)

                probs = F.softmax(logits, dim=1)

                predicted = (probs[:,1,:,:] > 0.5).squeeze().float()
                
                # 计算并记录单独的损失项
                loss_dict = compute_loss_val(predicted, patterns)
                val_loss += loss_dict["total_loss"].item() * spectra.size(0)
                val_pixel_loss += loss_dict["pixel_loss"].item() * spectra.size(0)
                val_boundary_loss += loss_dict["boundary_loss"].item() * spectra.size(0)
                val_relative_position_loss += loss_dict["relative_position_loss"].item() * spectra.size(0)


        
        val_loss /= len(test_data_loader.dataset)
        val_pixel_loss /= len(test_data_loader.dataset)
        val_boundary_loss /= len(test_data_loader.dataset)
        val_relative_position_loss /= len(test_data_loader.dataset)

        # 记录每种损失
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_total_loss": val_loss,
            "val_pixel_loss": val_pixel_loss,
            "val_boundary_loss": val_boundary_loss,
            "val_relative_position_loss": val_relative_position_loss,
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
    


