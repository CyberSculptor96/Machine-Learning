import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
# from torchsummary import summary

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
input_dims = 28 * 28
latent_dims = 20
batch_size = 64
epochs = 10


# VAE模型定义
class VAE(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(VAE, self).__init__()
        # 定义编码器的第一层，将输入数据映射到隐藏层
        self.fc1 = nn.Linear(input_dims, 512)
        # 定义编码器的第二层，分别输出均值向量和对数方差向量
        self.fc2_mean = nn.Linear(512, latent_dims)  # 均值向量
        self.fc2_logvar = nn.Linear(512, latent_dims)  # 对数方差向量
        # 定义解码器的第一层，从隐空间映射回隐藏层
        self.fc3 = nn.Linear(latent_dims, 512)
        # 定义解码器的第二层，从隐藏层映射到输出，即重建的输入数据
        self.fc4 = nn.Linear(512, input_dims)

    def encode(self, x):
        # 编码器的前向传播过程，使用ReLU激活函数
        h1 = F.relu(self.fc1(x))
        # 返回编码后的均值和对数方差向量
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    # 重参数化技巧，以便进行反向传播
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成标准正态分布随机数
        return mean + eps * std  # 返回重参数化后的样本点

    def decode(self, z):
        # 解码器的前向传播过程，使用ReLU激活函数
        h3 = F.relu(self.fc3(z))
        # 使用Sigmoid激活函数将输出限制在(0, 1)之间
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # VAE的前向传播过程
        mean, logvar = self.encode(x.view(-1, 784))  # 编码输入数据并获取均值和对数方差
        z = self.reparameterize(mean, logvar)  # 重参数化以获得潜在变量
        return self.decode(z), mean, logvar  # 返回重建数据、均值和对数方差


# 损失函数
def vae_loss(recon_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD


# 训练函数
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    loss_history = []  # 用于记录每个epoch的loss

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(data)
            loss = criterion(recon_batch, data, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        loss_history.append(train_loss / len(dataloader.dataset))  # 记录当前epoch的loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}')
        if (epoch + 1) % 10 == 0:  # 每10个epoch可视化一次
            visualize_reconstruction(model, dataloader, device)

    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def visualize_reconstruction(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        # 随机选择一些图片进行可视化
        sample = next(iter(dataloader))[0][:16]
        sample = sample.to(device)
        recon, _, _ = model(sample)
        # 将图片移回cpu
        sample = sample.cpu()
        recon = recon.cpu()
        # 绘制原始图像
        imshow(make_grid(sample.view(sample.size(0), 1, 28, 28), nrow=4), "Original Images")
        # 绘制重构图像
        imshow(make_grid(recon.view(recon.size(0), 1, 28, 28), nrow=4), "Reconstructed Images")


from sklearn.decomposition import PCA

# 隐空间可视化函数
def visualize_latent_space(model, dataloader, device, n_components=2):
    model.eval()
    with torch.no_grad():
        # 从数据集中获取一批图像
        images, labels = next(iter(dataloader))
        images = images.view(images.size(0), -1).to(device)
        mean, logvar = model.encode(images)
        z = model.reparameterize(mean, logvar)
        encoded_imgs = z.cpu().numpy()
        labels = labels.numpy()

        # 使用PCA进行降维
        pca = PCA(n_components=n_components)
        encoded_imgs_reduced = pca.fit_transform(encoded_imgs)

        # 可视化2D隐空间
        plt.figure(figsize=(10, 8))
        for i in range(10):
            indices = labels == i
            plt.scatter(encoded_imgs_reduced[indices, 0], encoded_imgs_reduced[indices, 1], label=f'Digit {i}')

        plt.legend()
        plt.title('2D Latent Space')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.show()

# 显示图像
def imshow(img, title):
    npimg = img.cpu().numpy()
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# 隐空间插值
def interpolate_latent_space(model, dataloader, device, start_img, end_img, n_interpolations=10):
    model.eval()
    with torch.no_grad():
        # 将选定的图像编码到隐空间
        start_data = start_img.to(device).view(-1, 784)  # 确保输入的形状是正确的
        end_data = end_img.to(device).view(-1, 784)  # 确保输入的形状是正确的
        _, start_latent, _ = model(start_data)
        _, end_latent, _ = model(end_data)

        # 在隐空间中线性插值
        interpolation_weights = torch.linspace(0, 1, n_interpolations).view(-1, 1).to(device)
        latent_interpolation = start_latent + (end_latent - start_latent) * interpolation_weights

        # 解码插值的隐向量
        interpolated_images = model.decode(latent_interpolation)

        # 将插值后的图像显示为网格
        grid = make_grid(interpolated_images.view(n_interpolations, 1, 28, 28), nrow=n_interpolations)
        imshow(grid, "Interpolated Latent Space")

# 主函数
def main():
    # 数据加载
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # 实例化模型和优化器
    model = VAE(input_dims, latent_dims).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 可选：展示模型的结构信息
    # summary(model, input_size=(1, 28*28))

    # 训练模型
    train(model, train_loader, vae_loss, optimizer, epochs)

    # 可视化重构图像
    visualize_reconstruction(model, train_loader, device)

    # 可视化隐空间
    visualize_latent_space(model, train_loader, device, n_components=2)

    # 进行潜在空间插值
    # 从MNIST数据集中选择两个随机的图像
    dataloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=2,
                            shuffle=True)
    start_img, end_img = next(iter(dataloader))[0][:2]
    interpolate_latent_space(model, dataloader, device, start_img, end_img, n_interpolations=10)

    # 保存模型参数
    save = False
    if save == True:
        torch.save(model.state_dict(), "vae.pth")

    import shutil, os
    shutil.rmtree('data') if os.path.exists('data') else None

    
if __name__ == "__main__":
    main()