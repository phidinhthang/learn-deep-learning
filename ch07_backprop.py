# ch07_backprop.py
# HỌC trọng số TỰ ĐỘNG để giải XOR — bằng lan truyền ngược (backpropagation).
# Khác Chương 6 (ta ĐẶT TAY trọng số vì đã biết đáp án), ở đây mạng tự mò ra
# từ khởi tạo ngẫu nhiên, chỉ nhờ dữ liệu + quy tắc: forward -> loss -> backward -> update.
# Cách chạy:  python ch07_backprop.py

import numpy as np
from ddl.numpy_nn import Linear, ReLU, Sequential, SoftmaxCrossEntropy, sgd_step

np.random.seed(42)   # tái lập kết quả (reproducibility)

# Dữ liệu XOR — giờ coi là bài PHÂN LOẠI 2 lớp: nhãn 0 hoặc 1
X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([0, 1, 1, 0])            # nhãn lớp (class labels), shape (4,)

# Mạng: 2 đầu vào -> 8 nơ-ron ẩn (ReLU) -> 2 logits (điểm thô cho 2 lớp)
net = Sequential([
    Linear(2, 8, seed=42),            # tầng ẩn
    ReLU(),
    Linear(8, 2, seed=43),            # tầng ra: 2 logits cho lớp 0 và lớp 1
])
loss_fn = SoftmaxCrossEntropy()
lr, n_epochs = 1.0, 2000

# Vòng lặp huấn luyện (training loop) — y hệt khung Chương 4–5, chỉ là mạng sâu hơn
for epoch in range(1, n_epochs + 1):
    logits = net.forward(X)               # (1) forward: tính logits
    loss = loss_fn.forward(logits, y)     # (2) đo mất mát
    grad = loss_fn.backward()             # (3) dJ/dlogits = (p - onehot(y)) / N
    net.backward(grad)                    # (4) lan truyền ngược: điền dW, db cho MỌI tầng
    sgd_step(net.parameters(), lr)        # (5) cập nhật: theta <- theta - lr * grad
    if epoch == 1 or epoch % 400 == 0:
        print(f"epoch {epoch:4d} | loss = {loss:.4f}")

# Dự đoán cuối: nhãn = lớp có logit lớn nhất (argmax)
logits = net.forward(X)
pred = np.argmax(logits, axis=1)
print("Dự đoán   =", pred)
print("XOR đúng  =", y)
print("Độ chính xác (accuracy) =", np.mean(pred == y))
