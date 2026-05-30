"""ddl/numpy_nn.py — một mạng nơ-ron tí hon viết tay bằng NumPy.

Xây dựng qua HAI chương:
  - Chương 6 (file này): phần LƯỢT TÍNH XUÔI (forward pass) — Linear, ReLU, Sequential.
  - Chương 7: thêm phần LAN TRUYỀN NGƯỢC (backward) + parameters() + sgd_step + SoftmaxCrossEntropy.

Cấu trúc các lớp (layer) ở đây cố tình bắt chước PyTorch, để khi sang Chương 9–10
bạn thấy PyTorch chỉ là tự động hóa đúng những thứ ta tự làm ở đây.
"""

import numpy as np


class Linear:
    """Tầng tuyến tính (fully-connected layer): y = x @ W + b.

    Quy ước shape (xem Hợp đồng khóa học §4.2):
        W: (in_features, out_features),  b: (out_features,)
        x: (N, in_features)  ->  y: (N, out_features)
    """

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # Khởi tạo trọng số bằng số ngẫu nhiên NHỎ (kiểu He). Vì sao không khởi tạo
        # toàn số 0? Nếu mọi trọng số bằng nhau, các nơ-ron sẽ học y hệt nhau và
        # không bao giờ tách ra — gọi là "phá vỡ đối xứng" (symmetry breaking). Chi
        # tiết hơn ở Chương 7–8.
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None   # nhớ lại input để Chương 7 dùng khi tính backward

    def forward(self, x):
        self.x = x                  # cache input
        return x @ self.W + self.b  # phép tuyến tính

    # backward(grad) và parameters() sẽ được thêm ở Chương 7.


class ReLU:
    """Hàm kích hoạt ReLU: f(x) = max(0, x). Áp theo từng phần tử (element-wise)."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)         # nhớ chỗ nào dương (cần cho backward ở Chương 7)
        return x * self.mask        # giữ phần dương, ép phần âm về 0

    # backward(grad) sẽ được thêm ở Chương 7.


class Sequential:
    """Xâu chuỗi nhiều tầng lại: forward đi lần lượt qua từng tầng."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:   # đưa dữ liệu qua từng tầng theo thứ tự
            x = layer.forward(x)
        return x

    # backward(grad) và parameters() sẽ được thêm ở Chương 7.
