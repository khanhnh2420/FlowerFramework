import flwr as fl
import torch
from utils import save_complete_model  # Import hàm lưu mô hình từ utils.py
from cnn_model import CNNModel  # Import mô hình CNN từ file cnn_model.py

# Hàm tính toán trung bình có trọng số cho các chỉ số (accuracy)
def weighted_average(metrics):
    """Tính trung bình có trọng số cho các chỉ số."""
    accuracies = []
    examples = []
    
    # Duyệt qua các metrics từ mỗi client
    for num_examples, m in metrics:
        if "accuracy" in m:  # Kiểm tra xem có 'accuracy' trong metrics không
            accuracies.append(num_examples * m["accuracy"])
            examples.append(num_examples)
    
    # Nếu có các chỉ số hợp lệ, tính trung bình có trọng số
    if examples:
        return {"accuracy": sum(accuracies) / sum(examples)}
    else:
        return {"accuracy": 0.0}  # Trả về 0 nếu không có chỉ số 'accuracy' hợp lệ

# Khởi tạo mô hình
def init_model():
    model = CNNModel()  # Tạo mô hình CNN
    return model

# Hàm callback để lưu mô hình sau mỗi vòng huấn luyện
def save_model_callback(model, round_num):
    """Lưu mô hình toàn cục sau mỗi vòng huấn luyện."""
    filename = f"complete_model_round_{round_num}.pth"
    save_complete_model(model, subdirectory="models", filename=filename)
    print(f"Model for round {round_num} saved successfully.")

# Hàm này dùng để khởi động serve
def start_flower_server():
    # Khởi tạo mô hình toàn cục
    model = init_model()
    
    # Cấu hình server
    num_rounds = 2  # Đặt số vòng huấn luyện ở đây

    # Tạo chiến lược FedAvg với hàm tổng hợp metrics cho đánh giá và huấn luyện
    strategy = fl.server.strategy.FedAvg(
        fit_metrics_aggregation_fn=weighted_average,  # Hàm tổng hợp độ chính xác cho các chỉ số fit
        evaluate_metrics_aggregation_fn=weighted_average,  # Hàm tổng hợp độ chính xác cho các chỉ số evaluate
        
    )

    # Khởi động Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Địa chỉ server
        config={"num_rounds": num_rounds},  # Số vòng huấn luyện
        strategy=strategy,  # Chiến lược huấn luyện
    )

    # Lưu mô hình sau mỗi vòng huấn luyện
    for round_num in range(1, num_rounds + 1):
        save_model_callback(model, round_num)

if __name__ == "__main__":
    start_flower_server()

