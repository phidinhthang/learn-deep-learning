# ch04_linear_regression.py
# Hồi quy tuyến tính (linear regression) TỪ SỐ 0: mô hình + MSE + gradient descent,
# tất cả viết tay bằng NumPy. Đây là mô hình BIẾT TỰ HỌC đầu tiên của khóa học.
# Cách chạy:  python ch04_linear_regression.py

import numpy as np
from ddl.preprocessing import Standardizer

np.random.seed(42)   # cố định seed để kết quả lặp lại được

# ===== (1) Dữ liệu =====
# Cố ý dùng quan hệ tuyến tính HOÀN HẢO y = 2x để dễ kiểm chứng kết quả.
# Hình dung: x = "diện tích nhà", y = "giá nhà".
x = np.array([1.0, 2.0, 3.0, 4.0])     # đặc trưng (feature)
y = np.array([2.0, 4.0, 6.0, 8.0])     # mục tiêu (target)
N = len(x)

# ===== (2) Tiền xử lý: chuẩn hóa đặc trưng =====
# Đưa x về trung bình 0, độ lệch chuẩn 1 -> giúp gradient descent học nhanh, ổn định.
scaler = Standardizer()
x_std = scaler.fit_transform(x)
print("x sau chuẩn hóa:", np.round(x_std, 4))

# ===== (3) Khởi tạo tham số (parameters) =====
w = 0.0          # trọng số (weight) = độ dốc của đường thẳng
b = 0.0          # độ chệch (bias)   = giao điểm với trục tung
lr = 0.1         # tốc độ học (learning rate)
n_steps = 50     # số bước cập nhật

# ===== (4) Vòng lặp huấn luyện: forward -> loss -> gradient -> update =====
for step in range(n_steps + 1):
    y_pred = w * x_std + b                  # forward: dự đoán y_hat = w*x + b
    loss = np.mean((y_pred - y) ** 2)       # MSE: trung bình bình phương sai số

    if step % 10 == 0:                      # in tiến trình mỗi 10 bước
        print(f"step {step:3d} | loss = {loss:.6f} | w = {w:.4f}, b = {b:.4f}")

    # Gradient của MSE (suy ra từ quy tắc chuỗi — xem phần Toán của chương):
    residual = y_pred - y                   # r = y_hat - y  (phần dư)
    grad_w = 2 * np.mean(residual * x_std)  # dJ/dw = (2/N) * sum(r * x)
    grad_b = 2 * np.mean(residual)          # dJ/db = (2/N) * sum(r)

    # Cập nhật tham số theo gradient descent: theta <- theta - lr * grad
    w -= lr * grad_w
    b -= lr * grad_b

# ===== (5) Kết quả học được =====
print(f"Học xong: w = {w:.4f}, b = {b:.4f}")

# ===== (6) Suy luận (inference) trên dữ liệu MỚI =====
# Lưu ý: dùng scaler.transform (KHÔNG fit lại) để áp ĐÚNG mean/std đã học từ train.
x_new = np.array([5.0])
x_new_std = scaler.transform(x_new)
y_new = w * x_new_std + b
print(f"Dự đoán cho x=5: {y_new[0]:.4f} (đúng phải là 10)")
