# ch05_logistic_regression.py
# Hồi quy logistic (logistic regression) TỪ SỐ 0: phân loại nhị phân (binary
# classification) bằng sigmoid + cross-entropy, vẫn viết tay bằng NumPy.
# Khung "model -> loss -> gradient -> update" GIỐNG HỆT Chương 4.
# Cách chạy:  python ch05_logistic_regression.py

import numpy as np
from ddl.functions import sigmoid, bce_loss

np.random.seed(42)

# ===== (1) Dữ liệu: 6 sinh viên =====
#   x = số giờ ôn thi CHÊNH LỆCH so với mốc 3.5 giờ  (x=0 nghĩa là học 3.5 giờ)
#   y = kết quả:  1 = đậu (pass),  0 = rớt (fail)
x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
N = len(x)

# ===== (2) Khởi tạo tham số =====
w, b = 0.0, 0.0
lr, n_steps = 0.5, 8

# ===== (3) Vòng lặp huấn luyện: forward -> loss -> gradient -> update =====
for step in range(n_steps + 1):
    z = w * x + b                                  # phần tuyến tính (giống Chương 4)
    p = sigmoid(z)                                 # ép z về xác suất p trong (0, 1)
    loss = bce_loss(p, y)                          # cross-entropy
    acc = np.mean((p >= 0.5) == (y == 1))          # độ chính xác (accuracy)
    print(f"step {step} | loss = {loss:.4f} | w = {w:.4f}, b = {b:.4f} | acc = {acc:.3f}")

    # Gradient của cross-entropy: rút gọn lại ĐÚNG dạng (p - y) như hồi quy tuyến tính!
    residual = p - y                               # r = p - y
    grad_w = np.mean(residual * x)                 # dJ/dw = (1/N) sum(r * x)
    grad_b = np.mean(residual)                     # dJ/db = (1/N) sum(r)

    w -= lr * grad_w                               # cập nhật theta <- theta - lr*grad
    b -= lr * grad_b

# ===== (4) Kết quả =====
print(f"Học xong: w = {w:.4f}, b = {b:.4f}")
final_acc = np.mean((sigmoid(w * x + b) >= 0.5) == (y == 1))
print(f"Độ chính xác trên tập huấn luyện: {final_acc:.3f}")
print("Xác suất dự đoán cho từng điểm:")
for xi, yi, pi in zip(x, y, sigmoid(w * x + b)):
    print(f"  x={xi:+.1f}, y={int(yi)} -> p(đậu)={pi:.4f}")

# ===== (5) Suy luận cho một sinh viên MỚI: học 3.7 giờ (x = 0.2) =====
x_new = 0.2
p_new = sigmoid(w * x_new + b)
ket_qua = "ĐẬU" if p_new >= 0.5 else "RỚT"
print(f"Sinh viên học 3.7 giờ (x=0.2): p(đậu) = {p_new:.4f} -> dự đoán {ket_qua}")
