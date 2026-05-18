import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 优化的CNN残差块
class CNN_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(CNN_block, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        # 第三层卷积
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于残差连接的降采样
        self.dropout = nn.Dropout(0.1)  # 添加dropout防止过拟合
        

    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu3(out)
        return out

# 优化的CNN编码器
class CNN_Net(nn.Module):
    def __init__(self, block, blocks_num, include_top=False):
        super(CNN_Net, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 初始通道数

        # 初始卷积层，类似ResNet设计
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四层残差块，逐步下采样
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel),
            )
        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.flatten(x)
        return x  # 输出 [B, 512, 13, 13] 当 include_top=False

# 注意力机制模块
class Attention(nn.Module):
    def __init__(self, hidden_size, feature_dim):
        super(Attention, self).__init__()
        self.hidden_proj = nn.Linear(hidden_size, feature_dim)  # 投影GRU隐藏状态
        self.feature_proj = nn.Linear(feature_dim, feature_dim)  # 投影CNN特征

    def forward(self, hidden, features):
        # hidden: [B, hidden_size]
        # features: [B, feature_dim, num_locations], e.g., [B, 512, 169]
        B, feature_dim, num_locations = features.size()

        # 投影隐藏状态
        proj_hidden = self.hidden_proj(hidden).unsqueeze(2)  # [B, feature_dim, 1]

        # 投影特征
        features_transposed = features.permute(0, 2, 1)  # [B, num_locations, feature_dim], e.g., [B, 169, 512]
        proj_features = self.feature_proj(features_transposed)  # [B, num_locations, feature_dim], e.g., [B, 169, 512]
        proj_features = proj_features.permute(0, 2, 1)  # [B, feature_dim, num_locations], e.g., [B, 512, 169]

        # 计算注意力能量
        energy = torch.tanh(proj_features + proj_hidden)  # [B, feature_dim, num_locations]
        attention_weights = F.softmax(energy.sum(dim=1), dim=1)  # [B, num_locations]

        # 计算上下文向量
        context = torch.bmm(features, attention_weights.unsqueeze(2)).squeeze(2)  # [B, feature_dim]
        return context

# 优化的Meta_CRNNAG_Net模型
class Meta_CRNNAG_Net(nn.Module):
    def __init__(self, seq_len=201):
        super().__init__()
        # 编码器：CNN网络，输出空间特征图
        self.CNNLayer = CNN_Net(CNN_block, [2, 2, 2, 2], include_top=False)  # 输出 [B, 512, 13, 13]
        self.seq_len = seq_len
        self.layers_num = 2  # GRU层数
        self.hidden_size = 256  # GRU隐藏状态维度
        self.feature_dim = 512  # CNN特征维度

        # 注意力机制
        self.attention = Attention(self.hidden_size, self.feature_dim)   #  feature.shape

        # 解码器：单向GRU
        self.embedding = nn.Linear(6, 6)  # 序列输入嵌入层
        self.gru = nn.GRU(input_size=6, hidden_size=self.hidden_size, num_layers=self.layers_num, 
                          dropout=0.2, batch_first=True, bidirectional=False)
        self.fc_out = nn.Linear(self.hidden_size + self.feature_dim, 6)  # 拼接后的全连接层

        # 初始化GRU隐藏状态
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_transform = nn.Linear(512, self.hidden_size)

    def forward(self, x, trg, teacher_forcing_ratio=0.8):
        # 编码器
        features = self.CNNLayer(x)  # [B, 512, 13, 13]
        B, C, H, W = features.size()
        features_flat = features.view(B, C, H*W)  # [B, 512, 169]

        # 初始化GRU隐藏状态
        pooled = self.avgpool(features).view(B, -1)  # [B, 512]
        hidden = self.hidden_transform(pooled).unsqueeze(0).repeat(self.layers_num, 1, 1)  # [layers, B, hidden_size]

        batch_size = trg.shape[0]
        trg_length = trg.shape[-1]
        trg = trg.permute(0, 2, 1)  # [B, seq_len, 6]

        outputs = torch.zeros(batch_size, trg_length, 6).to(x.device)
        input_seq = torch.zeros(batch_size, 1, 6, device=x.device)  # 起始输入

        # 自回归解码
        for t in range(self.seq_len):
            # GRU步骤
            rnn_output, hidden = self.gru(input_seq, hidden)  # [B, 1, hidden_size]
            rnn_output = rnn_output[:, -1, :]  # [B, hidden_size]


            # 注意力机制
            context = self.attention(rnn_output, features_flat)  # [B, feature_dim]

            # 拼接GRU输出和上下文向量
            combined = torch.cat((rnn_output, context), dim=1)  # [B, hidden_size + feature_dim]

            # 预测下一步
            output = self.fc_out(combined)  # [B, 6]
            outputs[:, t, :] = output

            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            input_seq = trg[:, t, :].unsqueeze(1) if teacher_force else output.unsqueeze(1)

        return outputs.permute(0, 2, 1)  # [B, 6, seq_len]

# 示例用法
if __name__ == "__main__":
    model = Meta_CRNNAG_Net(seq_len=201)
    x = torch.randn(2, 1, 400, 400)  # 示例输入 (batch, channels, H, W)
    trg = torch.randn(2, 6, 201)     # 示例目标序列
    output = model(x, trg)
    print(output.shape)  # 输出应为 [2, 6, 201]
    print(model)
