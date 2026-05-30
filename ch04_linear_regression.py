# ch04_linear_regression.py
# Hồi quy tuyến tính (linear regression) TỪ SỐ 0: mô hình + MSE + gradient descent,
# tất cả viết tay bằng NumPy. Đây là mô hình BIẾT TỰ HỌC đầu tiên của khóa học.
# Cách chạy:  python ch04_linear_regression.py

import numpy as np

np.random.seed(42)   # thói quen: cố định seed để mọi thứ lặp lại được

# ===== (1) Dữ liệu: 5 ngày quan sát của một xe bán nước =====
#   x = nhiệt độ ngày đó CHÊNH LỆCH so với mốc 25°C  (ví dụ x=2 nghĩa là 27°C)
#   y = số ly nước bán được trong ngày
# Quan hệ THẬT (mà ta CHƯA cho mô hình biết) là: y = 4*x + 20
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = np.array([12.0, 16.0, 20.0, 24.0, 28.0])
N = len(x)

# ===== (2) Khởi tạo tham số (parameters) — bắt đầu từ con số 0, mô hình chưa biết gì =====
w = 0.0          # trọng số (weight) = độ dốc của đường thẳng
b = 0.0          # độ chệch (bias)   = giá trị y khi x = 0
lr = 0.1         # tốc độ học (learning rate)
n_steps = 50     # số bước cập nhật

# ===== (3) Vòng lặp huấn luyện: forward -> loss -> gradient -> update =====
for step in range(n_steps + 1):
    y_pred = w * x + b                       # forward: dự đoán y_hat = w*x + b
    loss = np.mean((y_pred - y) ** 2)        # MSE: trung bình bình phương sai số

    if step % 10 == 0:                       # in tiến trình mỗi 10 bước
        print(f"step {step:3d} | loss = {loss:.6f} | w = {w:.4f}, b = {b:.4f}")

    # Gradient của MSE (suy ra từ quy tắc chuỗi — xem phần Toán của chương):
    residual = y_pred - y                    # r = y_hat - y  (phần dư)
    grad_w = 2 * np.mean(residual * x)       # dJ/dw = (2/N) * sum(r * x)
    grad_b = 2 * np.mean(residual)           # dJ/db = (2/N) * sum(r)

    # Cập nhật tham số theo gradient descent: theta <- theta - lr * grad
    w -= lr * grad_w
    b -= lr * grad_b

# ===== (4) Kết quả mô hình TỰ học được =====
print(f"Học xong: w = {w:.4f}, b = {b:.4f}  (quan hệ thật: w=4, b=20)")

# ===== (5) Suy luận (inference) cho một ngày MỚI =====
# Ngày 26.5°C, tức lệch +1.5 so với mốc 25°C. Mô hình chưa từng thấy điểm này.
x_new = np.array([1.5])
y_new = w * x_new + b
print(f"Dự đoán cho ngày 26.5°C (x=1.5): {y_new[0]:.4f} (đúng phải là 26)")
