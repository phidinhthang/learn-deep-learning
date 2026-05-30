# ch02_numpy_tour.py
# Một "tour" nhanh qua Python và NumPy — vừa đủ để học deep learning.
# Cách chạy:  python ch02_numpy_tour.py

import numpy as np

# Đặt hạt giống ngẫu nhiên (random seed) = 42 để kết quả lặp lại được (reproducible).
# Mọi chương trong khóa học đều dùng seed = 42.
np.random.seed(42)

print("=== 1) Python cơ bản (basics) ===")
# Biến (variable) và các kiểu dữ liệu (data types) cơ bản
n_features = 3          # số nguyên (int)
learning_rate = 0.1     # số thực (float)
name = "MNIST"          # chuỗi ký tự (string)
is_training = True      # luận lý (bool): True / False
print(n_features, learning_rate, name, is_training)

# Danh sách (list): một dãy phần tử có thứ tự, có thể thay đổi
scores = [0.2, 0.5, 0.9]
scores.append(0.95)                 # thêm phần tử vào cuối
print("scores =", scores, "| đầu =", scores[0], "| cuối =", scores[-1])  # -1 = phần tử cuối

# Vòng lặp for + enumerate: lấy cả chỉ số (index) lẫn giá trị (value)
for i, s in enumerate(scores):
    print(f"  scores[{i}] = {s}")   # f-string: chèn giá trị vào trong chuỗi

# Từ điển (dictionary): ánh xạ khóa -> giá trị (key -> value); ta sẽ dùng cho config
config = {"lr": 0.1, "epochs": 5}
print("lr =", config["lr"], "| các khóa (keys) =", list(config.keys()))

# Hàm (function): đóng gói một việc để tái sử dụng
def square(x):
    """Trả về bình phương của x."""   # đây là docstring (mô tả hàm)
    return x * x

print("square(4) =", square(4))

# Câu điều kiện (if / elif / else)
def sign(x):
    if x > 0:
        return "dương (positive)"
    elif x < 0:
        return "âm (negative)"
    else:
        return "không (zero)"

print("sign(-3) =", sign(-3))

# Lớp (class): khuôn để tạo ra các đối tượng (objects) có dữ liệu + hành vi.
# Ta cần hiểu class vì từ Chương 6, mỗi tầng (layer) của mạng sẽ là một class.
class Counter:
    def __init__(self, start=0):    # hàm khởi tạo, chạy khi tạo đối tượng
        self.count = start          # self = chính đối tượng này; lưu dữ liệu vào nó
    def increment(self):            # một phương thức (method)
        self.count += 1
        return self.count

c = Counter()                        # tạo một đối tượng Counter (start mặc định = 0)
c.increment()
c.increment()
print("counter =", c.count)          # đã tăng 2 lần -> 2

print()
print("=== 2) NumPy: mảng (arrays) ===")
# Tạo mảng 1 chiều (1D) từ một list
a = np.array([1.0, 2.0, 3.0])
print("a =", a, "| shape =", a.shape, "| dtype =", a.dtype)

# Ma trận (matrix) = mảng 2 chiều (2D array)
M = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
print("M shape =", M.shape, "| số chiều ndim =", M.ndim)   # (2, 3) và 2 chiều

# Vài cách tạo mảng hay dùng
print("zeros   :", np.zeros(3))             # mảng toàn số 0
print("ones    :", np.ones((2, 2)))         # ma trận 2x2 toàn số 1
print("arange  :", np.arange(0, 6, 2))      # từ 0 đến <6, bước 2 -> [0 2 4]
print("linspace:", np.linspace(0, 1, 5))    # 5 số cách đều từ 0 đến 1

print()
print("=== 3) Vector hóa (vectorization): tính trên cả mảng cùng lúc ===")
# Cộng/nhân theo từng phần tử (element-wise) — KHÔNG cần viết vòng lặp
print("a + 10 =", a + 10)
print("a * 2  =", a * 2)
print("a ** 2 =", a ** 2)
b = np.array([10.0, 20.0, 30.0])
print("a + b  =", a + b)    # cộng từng cặp phần tử tương ứng
print("a * b  =", a * b)    # nhân từng cặp (đây KHÔNG phải nhân ma trận)

print()
print("=== 4) Indexing & slicing (lấy phần tử / lát cắt) ===")
print("M[0]    =", M[0])      # hàng (row) đầu tiên
print("M[0, 1] =", M[0, 1])   # hàng 0, cột (column) 1
print("M[:, 0] =", M[:, 0])   # dấu ':' = "tất cả" -> toàn bộ cột 0
print("M[:, 1:] =\n", M[:, 1:])  # mọi hàng, từ cột 1 trở đi

print()
print("=== 5) Reshape (đổi hình dạng mà giữ nguyên dữ liệu) ===")
v = np.arange(6)             # [0 1 2 3 4 5], shape (6,)
print("v =", v, "| shape", v.shape)
print("reshape(2, 3) =\n", v.reshape(2, 3))
print("reshape(-1)  =", v.reshape(-1))   # -1 nghĩa là "tự suy ra" -> làm phẳng (flatten)

print()
print("=== 6) Thu gọn theo trục (reductions along an axis) ===")
# axis=0: gộp DỌC theo các hàng  -> còn lại 1 giá trị cho mỗi CỘT
# axis=1: gộp NGANG theo các cột -> còn lại 1 giá trị cho mỗi HÀNG
print("M.sum()        =", M.sum())          # tổng tất cả phần tử
print("M.sum(axis=0)  =", M.sum(axis=0))    # tổng theo từng cột
print("M.sum(axis=1)  =", M.sum(axis=1))    # tổng theo từng hàng
print("M.mean(axis=0) =", M.mean(axis=0))   # trung bình (mean) theo từng cột

print()
print("=== 7) Broadcasting: tự 'kéo giãn' mảng nhỏ cho khớp mảng lớn ===")
# Cộng một vector shape (3,) vào TỪNG hàng của ma trận shape (2, 3)
row = np.array([100.0, 200.0, 300.0])
print("M + row =\n", M + row)
# Trừ đi trung bình của mỗi cột — một thao tác tiền xử lý (preprocessing) rất hay gặp!
print("M - M.mean(axis=0) =\n", M - M.mean(axis=0))

print()
print("=== 8) Nhân ma trận (matrix multiplication) với toán tử @ ===")
# X: 2 mẫu (samples), mỗi mẫu 3 đặc trưng (features) -> shape (2, 3)
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
# W: trọng số (weights), 3 đầu vào -> 2 đầu ra -> shape (3, 2)
W = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
Z = X @ W                    # (2, 3) @ (3, 2) = (2, 2)
print("X @ W =\n", Z, "| shape =", Z.shape)

print()
print("=== 9) Vài hàm sẽ dùng nhiều về sau ===")
z = np.array([-1.0, 0.0, 2.0])
print("np.exp(z)       =", np.exp(z))           # e^z, dùng trong sigmoid/softmax
print("np.maximum(0, z)=", np.maximum(0, z))    # chính là ReLU = max(0, z) !
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
print("argmax(axis=1)  =", logits.argmax(axis=1))  # vị trí lớn nhất mỗi hàng = lớp dự đoán

print()
print("=== 10) Số ngẫu nhiên lặp lại được nhờ seed ===")
r = np.random.randn(3)       # 3 số từ phân phối chuẩn (normal distribution)
print("randn(3) =", r)       # vì seed=42, bạn sẽ nhận ĐÚNG các số này
