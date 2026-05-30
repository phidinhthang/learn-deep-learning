# ch06_forward_pass.py
# Vì sao cần MẠNG NHIỀU TẦNG + phi tuyến? Bài toán kinh điển: XOR.
# Một đường thẳng KHÔNG giải được XOR; một mạng 2 tầng + ReLU thì GIẢI ĐƯỢC.
# Ở đây ta ĐẶT TAY trọng số (Chương 7 sẽ HỌC chúng tự động).
# Cách chạy:  python ch06_forward_pass.py

import numpy as np
from ddl.numpy_nn import Linear, ReLU, Sequential

# Dữ liệu XOR: nhãn là 1 khi đúng MỘT trong hai đầu vào bằng 1
X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([0., 1., 1., 0.])

print("=== Một đơn vị tuyến tính KHÔNG giải được XOR ===")
# Thử đường thẳng đơn giản nhất: out = x1 + x2  (w=[1,1], b=0)
lin_out = X @ np.array([1., 1.]) + 0.0
print("x1 + x2       =", lin_out)     # [0 1 1 2] -> sai ở (1,1): cần 0 mà ra 2
print("XOR mong muốn =", y)

print()
print("=== Mạng 2 tầng + ReLU GIẢI ĐƯỢC XOR ===")
# Kiến trúc: 2 đầu vào -> 2 nơ-ron ẩn (hidden) + ReLU -> 1 đầu ra
net = Sequential([Linear(2, 2), ReLU(), Linear(2, 1)])

# Đặt tay bộ trọng số đã biết là giải được XOR:
#   h1 = ReLU(x1 + x2),  h2 = ReLU(x1 + x2 - 1),  out = h1 - 2*h2
net.layers[0].W = np.array([[1., 1.],
                            [1., 1.]])
net.layers[0].b = np.array([0., -1.])
net.layers[2].W = np.array([[1.],
                            [-2.]])
net.layers[2].b = np.array([0.])

# Nhìn vào tầng ẩn (hidden layer) sau ReLU để hiểu mạng "bẻ cong" không gian thế nào
hidden = ReLU().forward(X @ net.layers[0].W + net.layers[0].b)
print("Hidden sau ReLU:\n", hidden)

out = net.forward(X)
print("Đầu ra mạng   =", out.ravel())
print("XOR mong muốn =", y)
