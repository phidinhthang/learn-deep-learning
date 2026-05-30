"""ddl/numpy_nn.py — một mạng nơ-ron tí hon viết tay bằng NumPy.

Xây dựng qua HAI chương:
  - Chương 6: phần LƯỢT TÍNH XUÔI (forward pass) — Linear, ReLU, Sequential.
  - Chương 7 (bản này): thêm phần LAN TRUYỀN NGƯỢC (backward) + parameters()
    + sgd_step + SoftmaxCrossEntropy.

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
        # Khởi tạo kiểu He: số ngẫu nhiên nhỏ, co giãn theo sqrt(2/in_features). KHÔNG
        # khởi tạo toàn số 0 — nếu mọi trọng số bằng nhau, các nơ-ron nhận cùng gradient
        # và học y hệt nhau mãi mãi ("đối xứng" không bị phá vỡ — symmetry breaking).
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None        # nhớ input để dùng khi tính backward
        self.dW = None       # gradient của mất mát theo W: dJ/dW (điền ở backward)
        self.db = None       # gradient của mất mát theo b: dJ/db (điền ở backward)

    def forward(self, x):
        self.x = x                  # cache input cho backward
        return x @ self.W + self.b  # phép tuyến tính

    def backward(self, grad):
        # grad = dJ/dy, shape (N, out) — đạo hàm mất mát theo ĐẦU RA của tầng này.
        # Áp quy tắc chuỗi (chain rule) để ra gradient theo W, theo b, và theo ĐẦU VÀO x.
        self.dW = self.x.T @ grad           # dJ/dW = x^T @ grad        -> (in, out)
        self.db = grad.sum(axis=0)          # dJ/db = cộng dồn theo batch -> (out,)
        return grad @ self.W.T              # dJ/dx = grad @ W^T (đẩy ngược về tầng trước) -> (N, in)

    def parameters(self):
        # Trả về các cặp (tham số, gradient của nó) để sgd_step cập nhật.
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Hàm kích hoạt ReLU: f(x) = max(0, x). Áp theo từng phần tử (element-wise)."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)         # nhớ chỗ nào dương (đạo hàm ReLU = 1 ở đó, = 0 ở chỗ còn lại)
        return x * self.mask        # giữ phần dương, ép phần âm về 0

    def backward(self, grad):
        # Đạo hàm ReLU: 1 nếu input > 0, ngược lại 0. Gradient chỉ "chảy qua" ở chỗ dương.
        return grad * self.mask

    def parameters(self):
        return []                   # ReLU không có tham số học được


class Sequential:
    """Xâu chuỗi nhiều tầng: forward đi xuôi, backward đi ngược lại."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:           # đưa dữ liệu qua từng tầng theo thứ tự
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers): # lan truyền NGƯỢC: tầng cuối -> tầng đầu
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:           # gom tham số (và gradient) của mọi tầng lại
            params += layer.parameters()
        return params


class SoftmaxCrossEntropy:
    """Mất mát cho phân loại nhiều lớp: gộp softmax + cross-entropy làm một.

    forward(logits, y) -> mất mát trung bình (một con số).
        logits: (N, K) điểm thô từ tầng ra;  y: (N,) nhãn lớp dạng số nguyên 0..K-1.
    backward() -> gradient của mất mát theo logits, shape (N, K).

    Vì sao gộp chung? Đạo hàm của (softmax rồi cross-entropy) rút gọn cực đẹp thành
    (p - onehot(y)) / N — gộp lại để tận dụng điều đó và tránh sai số số học.
    """

    def __init__(self):
        self.p = None
        self.y = None

    def forward(self, logits, y):
        # softmax theo từng hàng; trừ max trước khi exp để ổn định số (numerical stability)
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        self.p = exp / exp.sum(axis=1, keepdims=True)   # xác suất mỗi lớp, shape (N, K)
        self.y = y
        N = logits.shape[0]
        # cross-entropy = -log(xác suất gán cho lớp ĐÚNG), rồi lấy trung bình toàn batch
        correct_logp = -np.log(self.p[np.arange(N), y] + 1e-12)
        return correct_logp.mean()

    def backward(self):
        # Phép màu softmax + cross-entropy: gradient theo logits gọn còn (p - onehot(y)) / N
        N = self.y.shape[0]
        grad = self.p.copy()
        grad[np.arange(N), self.y] -= 1.0       # trừ 1 ở đúng vị trí lớp thật
        return grad / N                         # chia N vì mất mát là trung bình theo batch


def sgd_step(params, lr):
    """Một bước hạ gradient ngẫu nhiên (stochastic gradient descent — SGD).

    params: danh sách các cặp (tham số, gradient) lấy từ model.parameters().
    lr: tốc độ học (learning rate, ký hiệu eta η trong sách).
    Cập nhật TẠI CHỖ (in-place): theta <- theta - lr * grad cho mọi tham số.
    """
    for param, grad in params:
        param -= lr * grad          # numpy mảng -> trừ tại chỗ, model thấy ngay giá trị mới
