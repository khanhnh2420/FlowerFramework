import io
import torch
from torchvision import transforms
from PIL import Image
from cnn_model import CNNModel
from flask import Flask, render_template, request, redirect, url_for

# Khởi tạo Flask
app = Flask(__name__)

# Định nghĩa các nhãn (label) cho các lớp dự đoán
LABELS = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# Thêm lớp CNNModel vào danh sách an toàn
torch.serialization.add_safe_globals([CNNModel])

# Tiền xử lý hình ảnh
def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Chuyển hình ảnh sang RGB nếu cần
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Lỗi khi tiền xử lý hình ảnh: {e}")

# Hàm dự đoán
def predict_image(model, image_bytes):
    try:
        image_tensor = transform_image(image_bytes)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence_score = probabilities[predicted_class].item()

        # Lấy nhãn tương ứng
        label = LABELS[predicted_class] if predicted_class < len(LABELS) else "Unknown"
        return label, confidence_score
    except Exception as e:
        raise ValueError(f"Lỗi khi dự đoán: {e}")

# Route trang chủ
@app.route('/')
def index():
    return render_template('index.html', predicted_class=None, confidence_score=None, error=None)

# Route xử lý file mô hình và hình ảnh tải lên
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'model' not in request.files or 'file' not in request.files:
            return render_template('index.html', predicted_class=None, confidence_score=None, error='Vui lòng tải lên cả file mô hình và ảnh!')

        model_file = request.files['model']
        image_file = request.files['file']

        if model_file.filename == '' or image_file.filename == '':
            return render_template('index.html', predicted_class=None, confidence_score=None, error='File tải lên không được để trống!')

        # Đọc và tải mô hình từ file .pth
        model_bytes = model_file.read()
        model = torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu'))

        model.eval()

        # Đọc ảnh từ file tải lên
        image_bytes = image_file.read()

        # Dự đoán lớp của ảnh
        predicted_class, confidence_score = predict_image(model, image_bytes)

        # Trả về kết quả lên giao diện
        return render_template('index.html', predicted_class=predicted_class, confidence_score=confidence_score, error=None)

    except Exception as e:
        return render_template('index.html', predicted_class=None, confidence_score=None, error=f"Lỗi: {e}")

if __name__ == '__main__':
    app.run(debug=True)

