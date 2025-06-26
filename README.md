# Diabetic-Retinopathy-Detection

Created a diabetic retinopathy detection system using MobileNetV2 pre-trained model with ~97% training accuracy.

[**Kaggle Dataset**](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)

---

## Mô tả dự án

Dự án này xây dựng một hệ thống nhận diện bệnh võng mạc tiểu đường từ ảnh võng mạc sử dụng deep learning. Mô hình được huấn luyện dựa trên kiến trúc MobileNetV2 với dữ liệu từ Kaggle, sau đó đóng gói thành ứng dụng web đơn giản bằng Streamlit.

## Các bước chính

1. **Tiền xử lý dữ liệu**  
   - Đọc và cân bằng lại dữ liệu bằng kỹ thuật under-sampling.
   - Chia tập dữ liệu thành train/validation.
   - Chuẩn hóa ảnh về kích thước 224x224 và scale pixel về [0, 1].

2. **Huấn luyện mô hình**  
   - Sử dụng MobileNetV2 (pretrained ImageNet) làm backbone.
   - Thêm các lớp fully connected, dropout và softmax cho phân loại 5 lớp.
   - Đóng băng các lớp của MobileNetV2 trong quá trình huấn luyện.
   - Huấn luyện mô hình với loss `sparse_categorical_crossentropy` và optimizer `adam`.

3. **Lưu mô hình**  
   - Mô hình sau khi huấn luyện được lưu lại dưới dạng file `.h5`.

4. **Triển khai ứng dụng web**  
   - Ứng dụng Streamlit cho phép người dùng upload ảnh võng mạc.
   - Ảnh được resize, chuẩn hóa và đưa vào mô hình để dự đoán.
   - Kết quả phân loại được hiển thị trực tiếp trên giao diện.

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. chạy ứng dụng

streamlit run app.py

### 3. Sử dụng

Truy cập giao diện web được mở ra.
Upload ảnh võng mạc (định dạng jpg, jpeg, png).
Xem kết quả phân loại:
1. No diabetic retinopathy
2. Mild diabetic retinopathy
3. Moderate diabetic retinopathy
4. Severe diabetic retinopathy
5. Proliferate diabetic retinopathy