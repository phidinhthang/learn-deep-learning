"""ddl/preprocessing.py — các công cụ tiền xử lý (preprocessing) dữ liệu.

Giới thiệu ở Chương 4. Hiện có:
  - Standardizer: chuẩn hóa dữ liệu về trung bình 0, độ lệch chuẩn 1.
"""

import numpy as np


class Standardizer:
    """Chuẩn hóa (standardize): đưa mỗi đặc trưng (feature) về trung bình 0, độ lệch chuẩn 1.

    Công thức:  x' = (x - mean) / std

    Quy trình dùng giống scikit-learn:
        scaler = Standardizer()
        X_train_std = scaler.fit_transform(X_train)   # HỌC mean/std từ tập train
        X_test_std  = scaler.transform(X_test)        # DÙNG LẠI mean/std đó cho test
    """

    def __init__(self):
        self.mean_ = None   # trung bình (mean) của mỗi đặc trưng, học từ dữ liệu
        self.std_ = None    # độ lệch chuẩn (standard deviation) của mỗi đặc trưng

    def fit(self, X):
        # HỌC (ghi nhớ) mean và std từ dữ liệu. axis=0 nghĩa là tính theo từng cột
        # (mỗi cột là một đặc trưng). Với mảng 1 chiều, kết quả là một số vô hướng.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)   # std của NumPy chia cho N (độ lệch chuẩn tổng thể)
        # Phòng trường hợp một đặc trưng không đổi (std = 0) sẽ gây chia cho 0:
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        # Áp công thức chuẩn hóa, DÙNG mean/std đã học ở fit().
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # Tiện lợi: fit (học) rồi transform (áp dụng) trong một bước.
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # Đảo ngược: từ giá trị đã chuẩn hóa quay về giá trị gốc.
        return X_scaled * self.std_ + self.mean_
