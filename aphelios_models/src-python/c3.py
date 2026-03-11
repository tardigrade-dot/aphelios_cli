import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from safetensors.torch import save_file

# 1. 定义与之前 Rust 重构一致的模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入: 1x28x28
        # padding=1 保证卷积后尺寸依然是 28x28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 经过两次 2x2 MaxPool，尺寸变为 7x7
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # -> 16x14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # -> 32x7x7
        x = x.view(x.size(0), -1)                  # 展平
        x = self.fc(x)
        return x

def train():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备数据 (归一化到 0-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 官方均值和标准差
    ])

    train_dataset = datasets.MNIST('/Users/larry/coderesp/aphelios_cli/test_data/mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. 训练循环
    model.train()
    for epoch in range(1, 6): # 训练 5 轮
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        
        print(f"Epoch {epoch} Average Loss: {total_loss / len(train_loader):.4f}")

    # 4. 导出权重为 Safetensors
    # 注意：需要安装 safetensors: pip install safetensors
    model.eval()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, "/Users/larry/coderesp/aphelios_cli/aphelios_models/output/mnist_cnn.safetensors")
    print("Model saved to mnist_cnn.safetensors")

def d():
    from torchvision import datasets
    import torch
    from PIL import Image

    # 加载测试集
    test_data = datasets.MNIST(root='/Users/larry/coderesp/aphelios_cli/test_data/mnist', train=False, download=True)
    img, label = test_data[0] # 获取第一张图
    img.save(f"mnist_test_{label}.png")
    print(f"Saved a real MNIST image of digit: {label}")
if __name__ == "__main__":
    # train()
    d()