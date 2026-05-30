"""ddl/functions.py — các hàm toán học dùng đi dùng lại trong khóa học.

Giới thiệu ở Chương 5. Tất cả viết bằng NumPy (chưa dùng PyTorch).
  - Hàm kích hoạt (activation): sigmoid, softmax, relu
  - Hàm mất mát (loss): mse_loss, bce_loss
"""

import numpy as np


def sigmoid(z):
    """Sigmoid: ép một số thực bất kỳ về khoảng (0, 1) -> đọc được như XÁC SUẤT.

    Công thức:  sigmoid(z) = 1 / (1 + e^{-z})
    """
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    """ReLU (Rectified Linear Unit):  relu(z) = max(0, z).  Dùng nhiều từ Chương 6."""
    return np.maximum(0.0, z)


def softmax(z):
    """Softmax: biến một vector điểm số (logits) thành phân phối xác suất (cộng lại = 1).

    z có shape (N, K) (N mẫu, K lớp). Trừ đi max theo mỗi hàng để ỔN ĐỊNH số học
    (numerical stability) — tránh e^{số lớn} bị tràn. Dùng từ Chương 6 trở đi.
    """
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def mse_loss(y_hat, y):
    """Sai số toàn phương trung bình (Mean Squared Error) — dùng cho hồi quy (Chương 4)."""
    return np.mean((y_hat - y) ** 2)


def bce_loss(p, y, eps=1e-12):
    """Cross-entropy nhị phân (Binary Cross-Entropy) — dùng cho phân loại 2 lớp (Chương 5).

    Công thức:  -mean( y*log(p) + (1-y)*log(1-p) )
    Ta kẹp (clip) p vào [eps, 1-eps] để không bao giờ tính log(0) (sẽ ra vô cực).
    """
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
