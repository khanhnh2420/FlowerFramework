import flwr as fl
import torch
from utils import save_complete_model  # Import hàm lưu mô hình từ utils.py
from cnn_model import CNNModel  # Import mô hình CNN từ file cnn_model.py

# Hàm tính toán trung bình có trọng số cho các chỉ số (accuracy)
def weighted_average(metrics):
    """Tính trung bình có trọng số cho các chỉ số."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Khởi tạo mô hình
def init_model():
    model = CNNModel()  # Tạo mô hình CNN (hoặc mô hình của bạn)
    return model

# Hàm callback để lưu mô hình sau mỗi vòng huấn luyện
def save_model_callback(model, round_num):
    """Lưu mô hình toàn cục sau mỗi vòng huấn luyện."""
    filename = f"complete_model_round_{round_num}.pth"
    save_complete_model(model, subdirectory="models", filename=filename)
    print(f"Model for round {round_num} saved successfully.")

# Hàm này dùng để khởi động server và chiến lược
def start_flower_server():
    # Khởi tạo mô hình toàn cục
    model = init_model()
    
    # Cấu hình server
    server_config = fl.server.ServerConfig(num_rounds=3)  # Đặt số vòng huấn luyện ở đây
    num_rounds = server_config.num_rounds  # Lấy số vòng huấn luyện từ cấu hình

    # Tạo chiến lược FedAvg với hàm tổng hợp metrics
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,  # Hàm tổng hợp độ chính xác
    )

    # Khởi động Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Địa chỉ server
        config=server_config,  # Số vòng huấn luyện
        strategy=strategy,  # Chiến lược huấn luyện
    )

    # Lưu mô hình sau mỗi vòng huấn luyện
    for round_num in range(1, num_rounds + 1):
        save_model_callback(model, round_num)


if __name__ == "__main__":
    start_flower_server()

