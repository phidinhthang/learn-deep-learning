"""ddl/preprocessing.py — chuẩn hóa đặc trưng (feature standardization).

Standardizer đưa mỗi đặc trưng (feature) về trung bình 0, độ lệch chuẩn 1:
    x' = (x - mean) / std
Vì sao cần? Khi các đặc trưng có thang đo rất khác nhau (vd: tuổi 0..100 và thu nhập
0..10^9), "cái bát" mất mát bị méo dẹt, gradient descent đi zíc-zắc và chậm. Chuẩn
hóa làm cái bát tròn lại -> học nhanh và ổn định hơn.

LƯU Ý: thống kê (mean, std) phải học TỪ TẬP TRAIN rồi áp y nguyên cho val/test —
không được "nhìn trộm" dữ liệu kiểm tra (tránh rò rỉ dữ liệu — data leakage).
"""

import numpy as np


class Standardizer:
    def fit(self, X):
        # Học thống kê theo TỪNG CỘT (mỗi đặc trưng một mean/std)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8    # +eps nhỏ để không bao giờ chia cho 0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)     # tiện: học rồi áp luôn, trên cùng tập train

    def inverse_transform(self, X):
        # Phép NGƯỢC: từ giá trị đã chuẩn hóa quay về thang đo gốc
        return X * self.std_ + self.mean_
