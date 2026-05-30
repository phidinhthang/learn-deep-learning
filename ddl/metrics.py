"""ddl/metrics.py — các phép đo chất lượng mô hình (NumPy)."""

import numpy as np


def predict_labels(logits):
    """Từ logits (N, K) -> nhãn dự đoán (N,): lấy lớp có điểm (logit) cao nhất."""
    return np.argmax(logits, axis=-1)


def accuracy(y_pred_labels, y_true):
    """Độ chính xác (accuracy): tỉ lệ đoán đúng, một số trong [0, 1].

    = (số mẫu đoán đúng) / (tổng số mẫu).
    """
    return np.mean(y_pred_labels == y_true)
