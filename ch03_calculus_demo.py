# ch03_calculus_demo.py
# Minh họa BẰNG SỐ bốn ý tưởng toán cốt lõi: đạo hàm, gradient,
# gradient descent, và quy tắc chuỗi (chain rule).
# Cách chạy:  python ch03_calculus_demo.py

import numpy as np


# ---------- 1) Đạo hàm bằng sai phân (numerical derivative) ----------
def numerical_derivative(f, x, h=1e-5):
    # Đạo hàm = độ dốc (slope) khi nhích x một lượng rất nhỏ h.
    # Dùng công thức đối xứng (central difference) để sai số nhỏ hơn.
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 2          # f(x) = x^2  ->  đạo hàm (derivative) f'(x) = 2x


x0 = 3.0
print("=== 1) Đạo hàm của f(x)=x^2 tại x=3 ===")
print(f"  numerical : {numerical_derivative(f, x0):.6f}")   # tính bằng số
print(f"  2x        : {2 * x0:.6f}")                        # tính bằng giải tích


# ---------- 2) Gradient cho hàm NHIỀU biến ----------
def g(w):
    # g(w) = w0^2 + w1^2 ; gradient (giải tích) = [2*w0, 2*w1]
    return w[0] ** 2 + w[1] ** 2


def numerical_gradient(func, w, h=1e-5):
    grad = np.zeros_like(w)                  # cùng shape với w, khởi tạo bằng 0
    for i in range(len(w)):
        w_plus = w.copy();  w_plus[i]  += h  # chỉ nhích RIÊNG biến thứ i lên
        w_minus = w.copy(); w_minus[i] -= h  # và xuống một chút
        # đạo hàm riêng (partial derivative) theo biến thứ i:
        grad[i] = (func(w_plus) - func(w_minus)) / (2 * h)
    return grad


w = np.array([1.0, 2.0])
print("=== 2) Gradient của g(w)=w0^2+w1^2 tại w=[1,2] ===")
print("  numerical :", np.round(numerical_gradient(g, w), 6))
print("  [2w0,2w1] :", np.array([2 * w[0], 2 * w[1]]))


# ---------- 3) Gradient descent: 'đi xuống dốc' để tìm cực tiểu ----------
# Tối thiểu hóa f(x)=x^2 (đáy nằm ở x=0). Quy tắc cập nhật: x <- x - lr * f'(x)
x = 4.0
lr = 0.1
print("=== 3) Gradient descent trên f(x)=x^2 (x0=4, lr=0.1) ===")
for step in range(6):
    grad = 2 * x                # f'(x) = 2x  (hướng dốc lên)
    x = x - lr * grad           # đi NGƯỢC hướng dốc lên => xuống dốc
    print(f"  step {step + 1}: x = {x:.4f}, f(x) = {x ** 2:.4f}")


# ---------- 4) Quy tắc chuỗi (chain rule) ----------
# y = (3x+1)^2. Đặt u = 3x+1  ->  y = u^2.
# dy/du = 2u ; du/dx = 3   =>   dy/dx = (2u)*(3) = 6*(3x+1)
def y(x):
    return (3 * x + 1) ** 2


x1 = 1.0
print("=== 4) Quy tắc chuỗi cho y=(3x+1)^2 tại x=1 ===")
print(f"  numerical : {numerical_derivative(y, x1):.6f}")
print(f"  6*(3x+1)  : {6 * (3 * x1 + 1):.6f}")
