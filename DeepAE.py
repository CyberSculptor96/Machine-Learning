import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# from torchsummary import summary

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 4)  # 输出的编码维度是4
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # 输出值在[0, 1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()  # 将模型设置为训练模式
    for epoch in range(num_epochs):  # 迭代整个数据集num_epochs次
        for data in dataloader:  # 从数据加载器中逐批次取数据
            img, _ = data  # 获取图像数据，忽略标签
            img = img.view(img.size(0), -1).to(device)  # 将图像数据展成一维向量

            optimizer.zero_grad()  	# 梯度清零
            output = model(img)  	# 前向传播，计算模型的输出
            loss = criterion(output, img)  # 计算重构图像和原图像之间的MSE损失函数
            loss.backward()  	# 反向传播，计算损失关于权重的梯度
            optimizer.step()  	# 根据计算的梯度更新模型的参数
        # 每个epoch结束后，打印loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch + 1) % 10 == 0:  # 每10个epoch执行一次可视化操作
            visualize_reconstruction(model, dataloader)  # 可视化重构图像和原图像


def visualize_reconstruction(model, dataloader):
    with torch.no_grad():
        # 随机选择一些图片进行可视化
        sample = next(iter(dataloader))[0][:16]
        sample = sample.view(sample.size(0), -1).to(device)
        output = model(sample)
        # 将图片移回cpu
        sample = sample.cpu()
        output = output.cpu()
        # 绘制原始图像
        imshow(torchvision.utils.make_grid(sample.view(sample.size(0), 1, 28, 28), nrow=4), "Original Images")
        # 绘制重构图像
        imshow(torchvision.utils.make_grid(output.view(output.size(0), 1, 28, 28), nrow=4), "Reconstructed Images")

from sklearn.decomposition import PCA


# 可视化隐空间
def visualize_latent_space(model, dataloader, n_components=2):
    model.eval()
    with torch.no_grad():
        # 从数据集中获取一批图像
        images, labels = next(iter(dataloader))
        images = images.view(images.size(0), -1).to(device)
        encoded_imgs = model.encoder(images).cpu().numpy()

        # 检查隐空间维度是否大于2
        if encoded_imgs.shape[1] > 2:
            # 使用PCA进行降维
            pca = PCA(n_components=n_components)
            encoded_imgs_reduced = pca.fit_transform(encoded_imgs)
        else:
            encoded_imgs_reduced = encoded_imgs

        # 可视化2D隐空间
        plt.figure(figsize=(10, 8))
        for i in range(10):
            indices = labels == i
            plt.scatter(encoded_imgs_reduced[indices, 0], encoded_imgs_reduced[indices, 1], label=f'Digit {i}')

        plt.legend()
        plt.title('2D Encoded Images Space')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.show()


def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def generate_new_images(decoder, num_images, latent_dim, device):
    # 生成随机噪声向量
    random_latent_vectors = torch.randn(num_images, latent_dim).to(device)

    # 将噪声向量通过解码器生成图片
    with torch.no_grad():
        new_images = decoder(random_latent_vectors)
        new_images = new_images.view(new_images.size(0), 1, 28, 28)
        new_images = new_images.cpu()

    # 可视化生成的图片
    imshow(torchvision.utils.make_grid(new_images, nrow=4), "Generated Images")


def main():
    # 实例化模型并移至相应设备
    autoencoder = Autoencoder().to(device)

    # 可选：展示模型的结构信息
    # summary(autoencoder, input_size=(1, 28*28))

    # 选择训练模式还是加载预训练模型
    isTrain = False
    if isTrain == False:
        autoencoder.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # 设置训练模型的参数
    num_epochs = 10  # 训练轮数

    # 开始训练模型
    train(autoencoder, train_loader, criterion, optimizer, num_epochs)

    # 保存模型参数
    save = False
    if save == True:
        torch.save(autoencoder.state_dict(), 'autoencoder_save.pth')

    # 可视化隐空间
    visualize_latent_space(autoencoder, train_loader)

    # 生成新的图片
    generate_new_images(autoencoder.decoder, num_images=16, latent_dim=4, device=device)

    import shutil, os
    shutil.rmtree('data') if os.path.exists('data') else None

    
if __name__ == "__main__":
    main()