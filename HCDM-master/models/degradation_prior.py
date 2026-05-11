import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationPrior(nn.Module):
    """退化感知模块

    输入：低质合成图 (B,3,H,W) 和前景掩码 (B,1,H,W)
    输出：退化先验向量 (B,8)

    结构：
    1. 将输入按通道拼接
    2. 全局平均池化 + 全局最大池化
    3. 拼接池化结果
    4. 两层全连接 -> 输出8维
    """

    def __init__(self):
        super().__init__()

        # 输入维度: 2 * (3+1) = 8 (平均池化和最大池化各4维)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.relu = nn.ReLU()

    def forward(self, degraded_image, foreground_mask):
        """
        Args:
            degraded_image: (B, 3, H, W) 低质合成图
            foreground_mask: (B, 1, H, W) 前景掩码

        Returns:
            degradation_vector: (B, 8) 退化先验向量
        """
        # 按通道拼接输入
        x = torch.cat([degraded_image, foreground_mask], dim=1)  # (B, 4, H, W)

        # 全局平均池化
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 4)

        # 全局最大池化
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 4)

        # 拼接池化结果
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 8)

        # 两层全连接
        degradation_vector = self.relu(self.fc1(pooled))  # (B, 16)
        degradation_vector = self.fc2(degradation_vector)  # (B, 8)

        return degradation_vector


class FiLMLayer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 层

    用退化向量调制特征图
    """

    def __init__(self, feature_channels, degradation_dim=8):
        super().__init__()

        # 生成缩放因子 gamma 和偏移因子 beta
        self.gamma_layer = nn.Linear(degradation_dim, feature_channels)
        self.beta_layer = nn.Linear(degradation_dim, feature_channels)

    def forward(self, features, degradation_vector):
        """
        Args:
            features: (B, C, H, W) 特征图
            degradation_vector: (B, 8) 退化先验向量

        Returns:
            modulated_features: (B, C, H, W) 调制后的特征图
        """
        # 生成 gamma 和 beta
        gamma = self.gamma_layer(degradation_vector).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_layer(degradation_vector).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        # FiLM 调制: y = gamma * x + beta
        modulated_features = gamma * features + beta

        return modulated_features


if __name__ == "__main__":
    # 测试代码
    batch_size, channels, height, width = 2, 3, 64, 64

    # 创建测试数据
    degraded_image = torch.randn(batch_size, channels, height, width)
    foreground_mask = torch.randn(batch_size, 1, height, width)

    # 测试 DegradationPrior
    degradation_prior = DegradationPrior()
    degradation_vector = degradation_prior(degraded_image, foreground_mask)
    print(f"Degradation vector shape: {degradation_vector.shape}")  # 应该是 (2, 8)

    # 测试 FiLMLayer
    feature_channels = 32
    features = torch.randn(batch_size, feature_channels, height, width)
    film_layer = FiLMLayer(feature_channels)
    modulated_features = film_layer(features, degradation_vector)
    print(f"Modulated features shape: {modulated_features.shape}")  # 应该是 (2, 32, 64, 64)