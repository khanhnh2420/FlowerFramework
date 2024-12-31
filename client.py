import flwr as fl
import torch
from collections import OrderedDict
from torchvision import datasets, transforms
from cnn_model import CNNModel  # Mô hình phân loại hình ảnh
from centralized import load_data, load_model, train, test

# Địa chỉ của Flower server
SERVER_ADDRESS = "192.168.255.135:8080"

# Hàm cập nhật tham số mô hình
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Tải mô hình và dữ liệu
net = CNNModel()  # Sử dụng mô hình CNN
trainloader, testloader = load_data()

# Lớp Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        #print(f"Sending parameters: {params}")  # Log các trọng số gửi đi
        return params
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        print("Starting training...")
        train(net, trainloader, epochs=10)
        print("Training completed.")
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        print(f"Evaluating model: loss={loss}, accuracy={accuracy}")  # Log kết quả đánh giá
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    print(f"Starting Flower client and connecting to server at {SERVER_ADDRESS}...")
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=FlowerClient().to_client()
    )
    print("Flower client has stopped.")

