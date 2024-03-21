import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# from torchsummary import summary

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 卷积自编码器定义
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 卷积层编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 输出形状: [batch_size, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 输出形状: [batch_size, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # 输出形状: [batch_size, 64, 2, 2]
        )
        # 隐空间的维度为 [batch_size, 64, 2, 2]，即隐空间共有64x2x2=256个维度
        # 反卷积层解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 输出形状: [batch_size, 32, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 输出形状: [batch_size, 16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 输出形状: [batch_size, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 训练函数
def train(autoencoder, data_loader, criterion, optimizer, num_epochs):
    autoencoder.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in data_loader:
            imgs, _ = data
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = autoencoder(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(data_loader)
        loss_history.append(average_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss over epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 可视化函数
def visualize(autoencoder, data_loader):
    autoencoder.eval()
    images, _ = next(iter(data_loader))

    # 显示原始图像
    print('Original Images')
    imshow(torchvision.utils.make_grid(images[:5]))

    # 显示重建图像
    print('Reconstructed Images')
    with torch.no_grad():
        recon = autoencoder(images.to(device))
    imshow(torchvision.utils.make_grid(recon[:5].cpu()))


def imshow(img):
    img = img / 2 + 0.5  # 非规范化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 主函数
def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集加载
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化自编码器
    autoencoder = ConvAutoencoder().to(device)

    # 可选：展示模型的结构信息
    # summary(autoencoder, input_size=(3, 32, 32))

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # 训练自编码器
    num_epochs = 10
    train(autoencoder, train_loader, criterion, optimizer, num_epochs)

    # 可视化结果
    visualize(autoencoder, train_loader)

    import shutil, os
    shutil.rmtree('data') if os.path.exists('data') else None
    

if __name__ == '__main__':
    main()