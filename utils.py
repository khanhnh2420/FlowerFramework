# utils.py
import torch
import os


# Hàm lưu trọng số mô hình
def save_model(model, filename="global_model.pth"):
    """Lưu trọng số mô hình vào một tệp."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Hàm tải trọng số mô hình
def load_model(model, filename="global_model.pth"):
    """Tải trọng số mô hình từ tệp."""
    model.load_state_dict(torch.load(filename))
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    print(f"Model loaded from {filename}")
    return model

# Hàm lưu toàn bộ mô hình 
def save_complete_model(model, subdirectory="models", filename="complete_model.pth"):
    """Lưu toàn bộ mô hình (kiến trúc và trọng số) vào thư mục con trong dự án."""
    # Lấy thư mục hiện tại của dự án
    current_directory = os.getcwd()
    
    # Đường dẫn đầy đủ đến thư mục con
    directory = os.path.join(current_directory, subdirectory)
    
    # Kiểm tra xem thư mục đã tồn tại chưa, nếu chưa thì tạo mới
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Xây dựng đường dẫn đầy đủ để lưu mô hình
    filepath = os.path.join(directory, filename)
    
    # Lưu mô hình
    torch.save(model, filepath)
    print(f"Complete model saved as {filepath}")

# Hàm tải toàn bộ mô hình
def load_complete_model(filename="complete_model.pth"):
    """Tải toàn bộ mô hình (kiến trúc và trọng số)."""
    model = torch.load(filename)
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    print(f"Complete model loaded from {filename}")
    return model


