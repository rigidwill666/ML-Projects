import torch
import os
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image  # 用于生成GIF

# 超参数设置
BATCH_SIZE = 64
N_EPOCHS = 10
N_STEPS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mnist_dataloader():
    """
    加载 MNIST 训练数据集并返回数据加载器。
    如果数据集未下载，则会自动下载。
    """
    train_data_path = "data/MNIST/raw"
    train_data_exists = os.path.exists(train_data_path) and any(os.listdir(train_data_path))
    dataset = torchvision.datasets.MNIST(
        root="data", train=True, download=not train_data_exists,
        transform=torchvision.transforms.ToTensor()
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def corrupt(x, amount):
    """
    根据给定的噪声比例 `amount` 对输入数据 `x` 进行损坏。
    通过将输入数据与随机噪声按比例混合来实现。
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

# ====================== 改进1：增强的UNet架构 ======================#
class ResBlock(nn.Module):
    """ 残差块 """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)
class ImprovedUNet(nn.Module):
    """ 改进的UNet：添加残差连接和注意力机制 """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            ResBlock(32),
            nn.MaxPool2d(2),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            ResBlock(64),
            nn.MaxPool2d(2),
            nn.SiLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            ResBlock(64),
            nn.MaxPool2d(2),
            nn.SiLU()
        )
        # 注意力层,让网络关注全局特征
        self.attn = nn.MultiheadAttention(64, 4)
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            ResBlock(64),
            nn.Upsample(scale_factor=2),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 5, padding=2),
            ResBlock(64),
            nn.Upsample(scale_factor=2),
            nn.SiLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(96, 32, 5, padding=2),
            ResBlock(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, out_channels, 5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)  # [B,32,14,14]
        x2 = self.down2(x1)  # [B,64,7,7]
        x3 = self.down3(x2)  # [B,64,3,3]

        # 注意力机制
        b, c, h, w = x3.shape
        x_attn = x3.view(b, c, -1).permute(0, 2, 1) # 变换维度，适应注意力输入 [B, 64, 3×3] -> [B, 9, 64]
        attn_out, _ = self.attn(x_attn, x_attn, x_attn) # 自注意力机制 [B, 9, 64]
        x3 = attn_out.permute(0, 2, 1).view(b, c, h, w) # 变换回原始形状 [B, 64, 3, 3]

        # Decoder
        x = self.up1(x3)  # [B,64,6,6]
        x = nn.functional.interpolate(x, size=x2.shape[2:])  # [B, 64, 6, 6] -> [B, 64, 7, 7]
        x = torch.cat([x, x2], dim=1)  # [B,128,7,7]
        x = self.up2(x)  # [B,64,14,14]
        x = nn.functional.interpolate(x, size=x1.shape[2:])  # [B,64,14,14]
        x = torch.cat([x, x1], dim=1)  # [B,96,14,14]
        x = self.up3(x)  # [B,1,28,28]
        return x

# ====================== 改进2：带噪声调度的迭代生成 ======================#
def cosine_noise_schedule(n_steps, s=0.008):  # s:控制噪声衰减速度
    """ 余弦噪声调度策略 """
    steps = torch.arange(n_steps)
    f = torch.cos((steps / n_steps + s) / (1 + s) * torch.pi / 2) ** 2
    return f / f[0]

def predict(net, dataloader):
    """
    对模型进行预测并可视化输入数据、损坏数据和预测结果。
    """
    x, _ = next(iter(dataloader))
    x = x[:8].to(DEVICE)
    # 使用更合适的噪声范围
    amount = torch.linspace(0.2, 0.8, x.shape[0]).to(DEVICE)  # 避免完全清晰和完全噪声的极端情况
    noised_x = corrupt(x, amount)
    with torch.no_grad():
        preds = net(noised_x).cpu()

    # 修改显示方式
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    grid_x = torchvision.utils.make_grid(x.cpu(), nrow=8, padding=2, normalize=True)
    grid_noised = torchvision.utils.make_grid(noised_x.cpu(), nrow=8, padding=2, normalize=True)
    grid_preds = torchvision.utils.make_grid(preds, nrow=8, padding=2, normalize=True)

    axs[0].set_title("Input data")
    axs[0].imshow(grid_x.permute(1, 2, 0).squeeze(), cmap="gray")
    axs[1].set_title("Corrupted data")
    axs[1].imshow(grid_noised.permute(1, 2, 0).squeeze(), cmap="gray")
    axs[2].set_title("Network Predictions")
    axs[2].imshow(grid_preds.permute(1, 2, 0).squeeze(), cmap="gray")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def iterative_generate(net, num_images=8, n_steps=50, img_size=28):
    """ 从纯噪声开始迭代生成 """
    x = torch.randn(num_images, 1, img_size, img_size).to(DEVICE)
    noise_levels = cosine_noise_schedule(n_steps, s=0.01)  # 调整噪声调度

    for t in reversed(range(n_steps)):
        current_noise = noise_levels[t] * torch.ones(x.shape[0]).to(DEVICE)
        noised_x = corrupt(x, current_noise)
        with torch.no_grad():
            pred = net(noised_x)
        # 减小每步更新的强度
        x = x + (pred - x) * noise_levels[t] * 0.5  # 添加缩放因子
    return x

# ====================== 改进3：训练过程优化 ======================#
def train_model(net, dataloader):
    """ 改进的训练流程 """
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.9)  # 添加学习率调度

    for epoch in range(N_EPOCHS):
        avg_loss = 0
        for x, _ in dataloader:
            x = x.to(DEVICE)
            # 使用更均匀的噪声分布
            noise_amount = torch.rand(x.shape[0]).to(DEVICE)
            noisy_x = corrupt(x, noise_amount)

            pred = net(noisy_x)
            loss = loss_fn(pred, x)

            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss += loss.item()

        scheduler.step()  # 更新学习率
        print(f"Epoch {epoch} | Loss: {avg_loss / len(dataloader):.4f}")
    return net

if __name__ == "__main__":
    # 获取数据加载器
    train_dataloader = get_mnist_dataloader()
    # 初始化模型
    model = ImprovedUNet().to(DEVICE)
    # 训练模型
    trained_model = train_model(model, train_dataloader)
    # 进行预测和可视化
    predict(trained_model, train_dataloader)
    # 生成新图像
    final_images = iterative_generate(trained_model, n_steps=N_STEPS)
    plt.figure(figsize=(10, 4))
    plt.title("Generated Images")
    grid = torchvision.utils.make_grid(final_images.cpu(), nrow=8, normalize=True)
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.axis('off')
    plt.show()