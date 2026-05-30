# ch04_plot_loss.py
# VẼ hàm mất mát (loss function) để THẤY nó có hình "cái bát", và gradient descent
# lăn dần xuống đáy như thế nào.
# Cách chạy:  python ch04_plot_loss.py   (sẽ lưu hình ra file ch04_loss.png)

import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu giống ch04_linear_regression.py
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = np.array([12.0, 16.0, 20.0, 24.0, 28.0])

# Loss thật ra phụ thuộc CẢ w và b (nên "cái bát" nằm trong không gian 3 chiều).
# Để dễ nhìn, ở đây ta CỐ ĐỊNH b = 20 (giá trị tốt nhất) và chỉ vẽ loss theo w
# -> được một đường cong 1 chiều hình chữ U.
b = 20.0


def loss_for_w(w):
    y_pred = w * x + b
    return np.mean((y_pred - y) ** 2)


# (A) Đường cong loss theo w trên một khoảng rộng -> hình "cái bát" (parabola)
w_grid = np.linspace(-2, 10, 100)
loss_grid = [loss_for_w(w) for w in w_grid]

# (B) Chạy gradient descent (chỉ với w) để lấy các điểm "lăn xuống dốc"
w = 0.0
lr = 0.1
w_steps, loss_steps = [], []
for _ in range(8):
    w_steps.append(w)
    loss_steps.append(loss_for_w(w))
    grad_w = 2 * np.mean((w * x + b - y) * x)   # dJ/dw
    w -= lr * grad_w                            # một bước đi xuống dốc

# (C) Vẽ tất cả lên một hình
plt.figure(figsize=(7, 5))
plt.plot(w_grid, loss_grid, label="loss J(w) — cái bát")
plt.plot(w_steps, loss_steps, color="red", alpha=0.4)
plt.scatter(w_steps, loss_steps, color="red", zorder=5,
            label="các bước gradient descent")
plt.axvline(4.0, linestyle="--", color="gray", label="đáy: w = 4")
plt.xlabel("w (trọng số)")
plt.ylabel("loss (MSE)")
plt.title("Hàm mất mát hình cái bát — gradient descent lăn xuống đáy")
plt.legend()
plt.savefig("ch04_loss.png", dpi=120, bbox_inches="tight")
print("Đã lưu hình vào ch04_loss.png")
