# ch05_plot_sigmoid.py
# VẼ đường cong sigmoid đã học, cùng dữ liệu và ranh giới quyết định (decision boundary).
# Cách chạy:  python ch05_plot_sigmoid.py   (lưu hình ra ch05_sigmoid.png)

import numpy as np
import matplotlib.pyplot as plt
from ddl.functions import sigmoid

# Dữ liệu giống ch05_logistic_regression.py
x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

# Tham số xấp xỉ sau khi học (xem output của ch05). Lấy w=1.4, b=0 để minh họa.
w, b = 1.4, 0.0

# Đường cong xác suất p = sigmoid(w*x + b) trên một dải x rộng
x_grid = np.linspace(-4, 4, 200)
p_grid = sigmoid(w * x_grid + b)

plt.figure(figsize=(7, 5))
plt.scatter(x, y, color="black", zorder=5, label="dữ liệu (0=rớt, 1=đậu)")
plt.plot(x_grid, p_grid, label="p = sigmoid(w·x + b)")
plt.axhline(0.5, linestyle=":", color="gray")                 # ngưỡng 0.5
plt.axvline(0.0, linestyle="--", color="red",
            label="ranh giới quyết định (p=0.5)")             # z=0 -> x=0
plt.xlabel("x (giờ ôn thi so với mốc 3.5)")
plt.ylabel("p(đậu)")
plt.title("Hồi quy logistic: sigmoid chia hai lớp tại x = 0")
plt.legend()
plt.savefig("ch05_sigmoid.png", dpi=120, bbox_inches="tight")
print("Đã lưu hình vào ch05_sigmoid.png")
