import torch
import os
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 超参数设置
BATCH_SIZE = 128
N_EPOCHS = 3
LEARNING_RATE = 1e-3
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

class BasicUNet(nn.Module):
    """
    一个简单的 UNet 模型，用于图像去噪任务。
    包含下采样层、上采样层和跳跃连接。
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2)
        ])
        self.activation = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.down_layers):
            x = self.activation(layer(x))
            if i < 2:
                skip_connections.append(x)
                x = self.downscale(x)

        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += skip_connections.pop()
            x = self.activation(layer(x))
        return x


def predict(net, dataloader):
    """
    对模型进行预测并可视化输入数据、损坏数据和预测结果。
    """
    x, _ = next(iter(dataloader))
    x = x[:8].to(DEVICE)
    amount = torch.linspace(0, 1, x.shape[0]).to(DEVICE)
    noised_x = corrupt(x, amount)
    with torch.no_grad():
        preds = net(noised_x).cpu()

    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title("Input data")
    axs[0].imshow(torchvision.utils.make_grid(x.cpu())[0].clip(0, 1), cmap="Greys")
    axs[1].set_title("Corrupted data")
    axs[1].imshow(torchvision.utils.make_grid(noised_x.cpu())[0].clip(0, 1), cmap="Greys")
    axs[2].set_title("Network Predictions")
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys")
    plt.show()

def train_model(net, dataloader):
    """
    训练 UNet 模型。
    """
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    losses = []

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for x, _ in dataloader:
            x = x.to(DEVICE)
            noise_amount = torch.rand(x.shape[0]).to(DEVICE)
            noisy_x = corrupt(x, noise_amount)
            pred = net(noisy_x)
            loss = loss_function(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.extend(epoch_losses)
        print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:.5f}")

    plt.plot(losses)
    plt.ylim(0, 0.1)
    plt.show()
    return net

if __name__ == "__main__":
    # 获取数据加载器
    train_dataloader = get_mnist_dataloader()
    # 初始化模型
    model = BasicUNet().to(DEVICE)
    # 训练模型
    trained_model = train_model(model, train_dataloader)
    # 进行预测
    predict(trained_model, train_dataloader)
