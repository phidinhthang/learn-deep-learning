# Chương 2 — Python & NumPy vừa đủ (Python & NumPy just enough)

## 0. Chúng ta đang ở đâu (Where we are)

Ở **Chương 1** ta đã có hai thứ: (1) tấm **bản đồ quy trình (pipeline map)** — thu thập → chia →
tiền xử lý → xây mô hình → mất mát → vòng lặp huấn luyện → đánh giá → lưu → suy luận; và (2) **môi
trường** đã cài xong.

Chương này **không** phải một ô riêng trên bản đồ — nó là **bộ đồ nghề (toolbox)** để viết được mọi ô.
Vì sao? Hãy nhìn lại bản đồ: các bước "tiền xử lý", "xây mô hình", "mất mát", "vòng lặp huấn luyện"
thực chất đều là **làm toán trên những mảng số (arrays of numbers)**. Công cụ để viết phần toán đó gọn
gàng và nhanh chính là **NumPy**. Nên trước khi học toán (Chương 3) và bắt tay vào mô hình đầu tiên
(Chương 4), ta cần thành thạo đúng phần Python + NumPy sẽ dùng — không hơn.

> Chương này cũng tạo file đầu tiên cho package code dùng chung của khóa học: `ddl/__init__.py`.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Tại sao cần NumPy, mà không chỉ dùng Python "thường"?

Giả sử bạn có 1.000.000 con số và muốn nhân đôi tất cả. Với Python thuần, bạn viết một **vòng lặp
(loop)** chạy qua từng số — vừa **dài dòng**, vừa **chậm**. Với NumPy, bạn viết `a * 2` cho **cả mảng**
cùng lúc — ngắn gọn và nhanh hơn hàng chục lần. Cách "tính trên cả mảng một phát" này gọi là **vector
hóa (vectorization)**, và nó có hai lợi ích:

1. **Nhanh:** NumPy chạy phép tính ở tầng C tối ưu, không phải vòng lặp Python chậm chạp.
2. **Giống công thức toán:** `z = X @ W + b` trong code trông gần như y hệt công thức trên giấy.

Cả khóa học, ta sẽ liên tục biến công thức toán thành một vài dòng NumPy/PyTorch như vậy.

### 1.2. Mảng (array) là gì, và vì sao "shape" quan trọng?

Hãy hình dung một **mảng (array)** như một **lưới số (grid of numbers)**:

- 1 chiều (1D) = một **vector** (một hàng số): `[1, 2, 3]`.
- 2 chiều (2D) = một **ma trận (matrix)** (bảng số có hàng và cột).
- nhiều chiều hơn = **tensor** (ví dụ một ảnh màu là tensor 3 chiều: cao × rộng × kênh màu).

**Hình dạng (shape)** là bộ kích thước của lưới theo từng chiều, ví dụ shape `(2, 3)` = 2 hàng, 3 cột.

> ⚠️ Ghi nhớ sớm: **lỗi phổ biến nhất** khi học deep learning là **lệch shape (shape mismatch)** — kiểu
> như cố nhân một ma trận `(2, 3)` với một ma trận `(2, 3)` (không hợp lệ). Vì vậy, thói quen quan
> trọng nhất bạn rèn ở chương này là: **luôn để ý shape của mọi mảng.**

### 1.3. Broadcasting — "kéo giãn" mảng nhỏ cho khớp mảng lớn

Thường ta muốn cộng một vector nhỏ vào mỗi hàng của một ma trận lớn (ví dụ: cộng **độ chệch (bias)**
vào mọi mẫu). NumPy cho phép làm điều đó tự động bằng **broadcasting**: nó "kéo giãn" (lặp lại) mảng nhỏ
cho vừa với mảng lớn, **mà không thực sự sao chép dữ liệu** (nên vẫn nhanh và tiết kiệm bộ nhớ). Trực
giác: `(2, 3)` + `(3,)` → vector `(3,)` được cộng vào **từng** hàng.

---

## 2. Toán học (The math) — luật về shape và phép nhân ma trận

Phần "toán" của chương này nhẹ nhàng: nó là **luật chơi của các mảng**. (Đại số tuyến tính và giải tích
sẽ được học kỹ ở Chương 3.)

### 2.1. Luật broadcasting (chính xác)

So khớp shape **từ phải sang trái**. Hai chiều là "tương thích" nếu **bằng nhau**, hoặc **một trong hai
bằng 1** (chiều bằng 1 sẽ được kéo giãn). Ví dụ:

- `(2, 3)` và `(3,)` → coi `(3,)` như `(1, 3)`; chiều đầu 2 vs 1 → kéo giãn → kết quả `(2, 3)`. ✅
- `(2, 3)` và `(2, 1)` → kéo giãn cột → `(2, 3)`. ✅
- `(2, 3)` và `(2,)` → so từ phải: 3 vs 2 → **không tương thích** → lỗi. ❌

### 2.2. Phép nhân ma trận (matrix multiplication), toán tử `@`

Nếu $X$ có shape $(N, D)$ và $W$ có shape $(D, M)$ thì tích $Z = X W$ có shape $(N, M)$. **Luật vàng:**
hai chiều **trong cùng** phải bằng nhau ($D = D$), và chúng "triệt tiêu" nhau, để lại hai chiều ngoài:

$$ (N, \underbrace{D) \cdot (D}_{\text{phải khớp}}, M) \;\to\; (N, M). $$

Phần tử ở hàng $i$, cột $j$ của $Z$ là **tích vô hướng (dot product)** của hàng $i$ trong $X$ với cột
$j$ trong $W$:

$$ Z_{ij} = \sum_{k=1}^{D} X_{ik}\, W_{kj}. $$

**Ví dụ tính tay (worked example).** Lấy đúng $X, W$ trong code bên dưới:

$$ X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\;(2,3), \quad
   W=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}\;(3,2). $$

Phần tử $Z_{11}$ (hàng 1 của $X$ · cột 1 của $W$): $1\cdot1 + 2\cdot0 + 3\cdot1 = 4$. Tính nốt:

$$ Z = \begin{bmatrix}4&5\\10&11\end{bmatrix}\;(2,2). $$

Hãy nhớ cảnh tượng này: **mỗi mẫu (một hàng của $X$) đi qua trọng số $W$ để cho ra đầu ra** — đó chính
là một **tầng tuyến tính (linear layer)**, viên gạch đầu tiên của mọi mạng nơ-ron (Chương 4 & 6).

---

## 3. Cài đặt (Implementation)

### 3.1. Tạo package `ddl/`

Tạo thư mục `ddl/` và bên trong một file `__init__.py`. File này (dù gần như trống) báo cho Python biết
`ddl` là một **package**, để các chương sau ta viết được `from ddl.preprocessing import Standardizer`.

```python
# ddl/__init__.py
"""ddl — package chứa code dùng chung của khóa học "Deep Learning từ số 0".

Đây là điểm khởi đầu (package marker): nhờ có file __init__.py, Python coi thư mục
"ddl" là một package và cho phép ta viết `from ddl.xxx import ...` ở các chương sau.
"""
```

### 3.2. "Tour" Python + NumPy — `ch02_numpy_tour.py`

Dưới đây là toàn bộ chương trình minh họa, chia thành 10 phần. Bạn có thể đọc từng phần, rồi ghép lại
thành một file tên `ch02_numpy_tour.py` để chạy (file này cũng đã có sẵn trong repo).

**Phần đầu — import và đặt seed:**

```python
# ch02_numpy_tour.py
import numpy as np

# Đặt hạt giống ngẫu nhiên (random seed) = 42 để kết quả lặp lại được (reproducible).
np.random.seed(42)
```

**(1) Python cơ bản** — biến, list, vòng lặp, dict, hàm, if/else, và class (ta cần class vì từ Chương 6
mỗi tầng của mạng sẽ là một class):

```python
n_features = 3          # số nguyên (int)
learning_rate = 0.1     # số thực (float)
name = "MNIST"          # chuỗi ký tự (string)
is_training = True       # luận lý (bool)

scores = [0.2, 0.5, 0.9]
scores.append(0.95)                 # thêm phần tử vào cuối list
for i, s in enumerate(scores):      # enumerate cho cả chỉ số i lẫn giá trị s
    print(f"  scores[{i}] = {s}")   # f-string

config = {"lr": 0.1, "epochs": 5}   # dict: khóa -> giá trị

def square(x):
    """Trả về bình phương của x."""
    return x * x

class Counter:
    def __init__(self, start=0):    # hàm khởi tạo, self = chính đối tượng
        self.count = start
    def increment(self):
        self.count += 1
        return self.count
```

> Đừng lo nếu class còn lạ — ta chỉ cần nhớ ý: `class` là cái khuôn, `__init__` chạy khi tạo đối tượng,
> và `self` là cách đối tượng tự tham chiếu đến dữ liệu của mình.

**(2)–(10)** là phần NumPy: tạo mảng, vector hóa, indexing/slicing, reshape, thu gọn theo trục (axis),
broadcasting, nhân ma trận `@`, vài hàm hay dùng (`np.exp`, `np.maximum`, `argmax`), và số ngẫu nhiên.
Toàn bộ những dòng này nằm trong file `ch02_numpy_tour.py` (đã chú thích kỹ bằng tiếng Việt). Hai đoạn
đáng chú ý nhất cho phần sau của khóa học:

```python
# Trừ trung bình mỗi cột — đây chính là một bước TIỀN XỬ LÝ (preprocessing) hay gặp (Chương 4):
M - M.mean(axis=0)

# np.maximum(0, z) chính là hàm kích hoạt ReLU = max(0, z) ta sẽ dùng ở Chương 6:
np.maximum(0, z)
```

---

## 4. Chạy thử (Run it)

Nhớ đã kích hoạt môi trường ảo `.venv` (xem Chương 1). Tại thư mục dự án:

```powershell
python ch02_numpy_tour.py
```

**Kết quả mong đợi (expected output)** — riêng phần số thực (float) thì **cách hiển thị/làm tròn có thể
hơi khác** trên máy bạn, đó là bình thường:

```
=== 1) Python cơ bản (basics) ===
3 0.1 MNIST True
scores = [0.2, 0.5, 0.9, 0.95] | đầu = 0.2 | cuối = 0.95
  scores[0] = 0.2
  scores[1] = 0.5
  scores[2] = 0.9
  scores[3] = 0.95
lr = 0.1 | các khóa (keys) = ['lr', 'epochs']
square(4) = 16
sign(-3) = âm (negative)
counter = 2

=== 2) NumPy: mảng (arrays) ===
a = [1. 2. 3.] | shape = (3,) | dtype = float64
M shape = (2, 3) | số chiều ndim = 2
zeros   : [0. 0. 0.]
ones    : [[1. 1.]
 [1. 1.]]
arange  : [0 2 4]
linspace: [0.   0.25 0.5  0.75 1.  ]

=== 3) Vector hóa (vectorization): tính trên cả mảng cùng lúc ===
a + 10 = [11. 12. 13.]
a * 2  = [2. 4. 6.]
a ** 2 = [1. 4. 9.]
a + b  = [11. 22. 33.]
a * b  = [10. 40. 90.]

=== 4) Indexing & slicing (lấy phần tử / lát cắt) ===
M[0]    = [1. 2. 3.]
M[0, 1] = 2.0
M[:, 0] = [1. 4.]
M[:, 1:] =
 [[2. 3.]
 [5. 6.]]

=== 5) Reshape (đổi hình dạng mà giữ nguyên dữ liệu) ===
v = [0 1 2 3 4 5] | shape (6,)
reshape(2, 3) =
 [[0 1 2]
 [3 4 5]]
reshape(-1)  = [0 1 2 3 4 5]

=== 6) Thu gọn theo trục (reductions along an axis) ===
M.sum()        = 21.0
M.sum(axis=0)  = [5. 7. 9.]
M.sum(axis=1)  = [ 6. 15.]
M.mean(axis=0) = [2.5 3.5 4.5]

=== 7) Broadcasting: tự 'kéo giãn' mảng nhỏ cho khớp mảng lớn ===
M + row =
 [[101. 202. 303.]
 [104. 205. 306.]]
M - M.mean(axis=0) =
 [[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]

=== 8) Nhân ma trận (matrix multiplication) với toán tử @ ===
X @ W =
 [[ 4.  5.]
 [10. 11.]] | shape = (2, 2)

=== 9) Vài hàm sẽ dùng nhiều về sau ===
np.exp(z)       = [0.36787944 1.         7.3890561 ]
np.maximum(0, z)= [0. 0. 2.]
argmax(axis=1)  = [0 1]

=== 10) Số ngẫu nhiên lặp lại được nhờ seed ===
randn(3) = [ 0.49671415 -0.1382643   0.64768854]
```

Đối chiếu vài dòng với phần toán ở mục 2: `X @ W` cho đúng `[[4, 5], [10, 11]]` như ta tính tay; và
`M.sum(axis=0) = [5 7 9]` đúng là tổng **theo từng cột**. Nếu các con số khớp, bạn đã nắm được công cụ.

---

## 5. Bài tập (Exercises)

1. **Đoán shape.** Không chạy code, hãy nói kết quả có hợp lệ không, và nếu có thì shape là bao nhiêu:
   (a) `(4, 3)` + `(3,)`; (b) `(4, 3)` + `(4,)`; (c) `(4, 3) @ (3, 2)`; (d) `(4, 3) @ (4, 3)`.
2. **Tính tay rồi kiểm chứng.** Cho `A = [[1, 2], [3, 4]]` và `B = [[1, 0], [1, 1]]`. Tính `A @ B`
   bằng tay, sau đó viết vài dòng NumPy để kiểm tra.
3. **Chuẩn hóa một cột (preview tiền xử lý).** Cho `X = np.array([[1.0], [2.0], [3.0], [4.0]])`. Hãy
   tính `X_centered = X - X.mean()` rồi `X_scaled = X_centered / X.std()`. In ra `X_scaled.mean()` và
   `X_scaled.std()` — bạn kỳ vọng chúng xấp xỉ bao nhiêu? (Đây đúng là việc Chương 4 sẽ làm.)
4. **Vòng lặp vs vector hóa.** Viết một hàm cộng 5 vào mọi phần tử của một mảng bằng **vòng lặp `for`**,
   rồi viết lại bằng **một dòng NumPy**. Xác nhận hai kết quả giống nhau.
5. **Từ logits ra nhãn.** Cho `logits = np.array([[0.1, 2.0, 0.3], [1.5, 0.2, 0.9]])`, mỗi hàng là điểm
   số cho 3 lớp. Dùng `argmax(axis=1)` để lấy lớp dự đoán cho mỗi hàng. Đáp án là gì?

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. (a) hợp lệ → `(4, 3)`; (b) **không** hợp lệ (so từ phải: 3 vs 4); (c) hợp lệ → `(4, 2)`; (d) **không**
   hợp lệ cho `@` (chiều trong 3 vs 4 không khớp — đó là phép nhân từng phần tử thì mới cần shape bằng nhau).
2. `A @ B = [[1*1+2*1, 1*0+2*1], [3*1+4*1, 3*0+4*1]] = [[3, 2], [7, 4]]`.
3. Sau khi chuẩn hóa, `mean ≈ 0` và `std ≈ 1`. Đó là mục đích của chuẩn hóa: đưa dữ liệu về quanh 0 với
   độ phân tán 1.
4. Vòng lặp: `out = a.copy(); for i in range(len(a)): out[i] += 5`. Vector hóa: `out = a + 5`.
5. `[1 0]` (hàng 1 lớn nhất ở vị trí 1; hàng 2 lớn nhất ở vị trí 0).

</details>

---

## 6. Tiếp theo (What's next)

Giờ bạn đã đọc và viết được NumPy. Ở **Chương 3 — Toán cần thiết**, ta sẽ ôn lại đúng phần toán dùng cho
deep learning: vector, ma trận, **đạo hàm (derivative)**, **gradient**, và **quy tắc chuỗi (chain
rule)** — nền tảng để Chương 4 dạy mô hình **tự học** bằng gradient descent.
