import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_model import CNNModel

# Hàm tải dữ liệu cho mô hình centralized
def load_data(batch_size=32, dataset='CIFAR10'):
    """
    Hàm tải dữ liệu cho mô hình centralized.
    Args:
        batch_size (int): Kích thước batch khi tải dữ liệu.
        dataset (str): Bộ dữ liệu sử dụng ('CIFAR10' hoặc 'MNIST').

    Returns:
        trainloader, testloader: DataLoader cho huấn luyện và kiểm tra.
    """

    # Các phép biến đổi cho dữ liệu đầu vào
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa hình ảnh
    ])
    
    if dataset == 'CIFAR10':
        # Tải bộ dữ liệu CIFAR-10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'MNIST':
        # Tải bộ dữ liệu MNIST
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported. Use 'CIFAR10' or 'MNIST'.")

    # Tạo DataLoader cho huấn luyện và kiểm tra
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# Hàm load mô hình CNN
def load_model():
    model = CNNModel()  # Khởi tạo mô hình CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Chuyển mô hình lên GPU nếu có
    return model

# Hàm huấn luyện mô hình
def train(model, trainloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

# Hàm kiểm tra độ chính xác của mô hình
def test(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    running_loss = 0.0  # Khởi tạo biến chạy để tính tổng tổn thất

    criterion = nn.CrossEntropyLoss()  # Đảm bảo sử dụng hàm mất mát phù hợp với bài toán phân loại

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Tính toán tổn thất (loss)
            running_loss += loss.item()  # Cộng dồn tổn thất trong batch

            _, predicted = torch.max(outputs, 1)  # Tìm dự đoán có xác suất cao nhất
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Tính độ chính xác
    average_loss = running_loss / len(testloader)  # Tính trung bình tổn thất

    print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")  # In ra kết quả

    return average_loss, accuracy  # Trả về cả tổn thất và độ chính xác


