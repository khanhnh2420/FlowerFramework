from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# Khởi tạo Flask
app = Flask(__name__)

# Tiền xử lý hình ảnh
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))  # Đọc hình ảnh từ bộ nhớ
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Hàm dự đoán
def predict_image(model, image_bytes):
    image_tensor = transform_image(image_bytes)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Route trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý file mô hình và hình ảnh tải lên
@app.route('/predict', methods=['POST'])
def predict():
    if 'model' not in request.files or 'file' not in request.files:
        return redirect(request.url)
    
    model_file = request.files['model']
    image_file = request.files['file']
    
    if model_file.filename == '' or image_file.filename == '':
        return redirect(request.url)
    
    if model_file and image_file:
        # Đọc mô hình từ file .pth
        model_bytes = model_file.read()
        model = torch.load(io.BytesIO(model_bytes))
        model.eval()
        
        # Đọc hình ảnh từ file
        image_bytes = image_file.read()
        
        # Dự đoán kết quả
        predicted_class = predict_image(model, image_bytes)
        
        # Trả về kết quả
        return render_template('index.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)

