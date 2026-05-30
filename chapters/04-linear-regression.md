# Chương 4 — Hồi quy tuyến tính từ số 0 (Linear regression from scratch)

## 0. Chúng ta đang ở đâu (Where we are)

- **Chương 1:** bản đồ quy trình (pipeline).
- **Chương 2:** công cụ NumPy.
- **Chương 3:** đạo hàm, gradient, **gradient descent**, quy tắc chuỗi.

Đây là chương **bước ngoặt**: lần đầu tiên ta ghép tất cả lại thành một **mô hình biết tự học**. Trên
bản đồ, một chương này chạm vào **rất nhiều ô cùng lúc** — lần đầu đi gần trọn một vòng pipeline trên
một bài toán tí hon:

> (3) **tiền xử lý** → (4) **xây mô hình** (đường thẳng) → (5) **hàm mất mát** (MSE) → (6) **vòng lặp
> huấn luyện** (gradient descent) → (9) **suy luận** (dự đoán cho dữ liệu mới).

(Ô (2) **chia dữ liệu** ta tạm để dành đến Chương 8, vì ở đây dữ liệu chỉ là vài điểm minh họa.)

Mọi thứ viết tay bằng NumPy — không thư viện deep learning nào cả — để bạn thấy rõ từng bánh răng.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Bài toán: dự đoán một con số (regression)

**Hồi quy (regression)** nghĩa là dự đoán một **con số liên tục**: giá nhà, nhiệt độ ngày mai, chiều cao…
(Khác với **phân loại — classification** ở Chương 5, vốn dự đoán một *nhãn*: chó/mèo, số 0–9.)

Ví dụ xuyên suốt: cho biết **diện tích** căn nhà ($x$), hãy dự đoán **giá** của nó ($y$).

### 1.2. Mô hình: một đường thẳng

Giả thuyết đơn giản nhất: giá tỉ lệ tuyến tính với diện tích. Tức mô hình là một **đường thẳng**:

$$ \hat{y} = w\,x + b. $$

- $w$ — **trọng số (weight)** — là **độ dốc (slope)**: $x$ tăng 1 thì $\hat{y}$ tăng $w$.
- $b$ — **độ chệch (bias)** — là **giao điểm với trục tung (intercept)**: giá trị $\hat{y}$ khi $x = 0$.

$w$ và $b$ chính là **tham số (parameters)** mà mô hình phải *học*. Học xong = tìm được đường thẳng đi
"vừa khít" qua đám dữ liệu.

### 1.3. Đo độ sai: hàm mất mát MSE

Làm sao biết một đường thẳng "khít" hay không? Ta cần **đo** độ sai. Với mỗi điểm, sai số là khoảng cách
giữa dự đoán $\hat{y}_i$ và giá trị thật $y_i$. Ta dùng **sai số toàn phương trung bình (Mean Squared
Error — MSE)**:

$$ J(w, b) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2. $$

Vì sao **bình phương**? (1) để sai số âm và dương không triệt tiêu nhau; (2) để **phạt nặng** những sai
lớn; (3) nó **trơn (smooth)** và dễ lấy đạo hàm — điều kiện sống còn để dùng gradient descent.

### 1.4. Kế hoạch & vì sao cần tiền xử lý

Kế hoạch khớp đúng Chương 3: coi $J(w,b)$ là hàm cần **tối thiểu hóa**, tính **gradient** của nó theo
$w, b$, rồi **gradient descent** để mô hình tự tìm $w, b$ tốt nhất.

Còn **tiền xử lý (preprocessing)** thì sao? Hãy hình dung hàm mất mát $J$ như một **cái bát (bowl)** trong
không gian $(w, b)$; gradient descent là quả bóng lăn xuống đáy. Nếu các đặc trưng có **thang đo rất khác
nhau** (ví dụ diện tích cỡ hàng nghìn, số phòng cỡ vài đơn vị), cái bát bị **kéo méo thành một thung lũng
dài và hẹp**; quả bóng lăn **zíc-zắc** rất chậm. **Chuẩn hóa (standardization)** đưa mọi đặc trưng về
cùng thang đo (trung bình 0, độ lệch chuẩn 1), nắn cái bát **tròn đều** trở lại, nên gradient descent
xuống đáy **nhanh và ổn định** hơn nhiều. Đó là lý do ta chuẩn hóa *trước* khi huấn luyện.

---

## 2. Toán học (The math)

### 2.1. Mô hình (một và nhiều đặc trưng)

Một đặc trưng: $\hat{y} = w x + b$. Nhiều đặc trưng ($D$ đặc trưng, gói trong vector $\mathbf{x}$):

$$ \hat{y} = \mathbf{w}^\top \mathbf{x} + b, \qquad \text{hay cho cả lô: } \hat{\mathbf{y}} = X\mathbf{w} + b, $$

với $X$ shape $(N, D)$, $\mathbf{w}$ shape $(D,)$, $\hat{\mathbf{y}}$ shape $(N,)$. (Chương này dùng
$D = 1$ để mọi thứ thật rõ; dạng vector sẽ trở lại ở Chương 6.)

### 2.2. Hàm mất mát

$$ J(w, b) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2, \qquad \hat{y}_i = w x_i + b. $$

### 2.3. Tính gradient bằng quy tắc chuỗi

Ta cần $\dfrac{\partial J}{\partial w}$ và $\dfrac{\partial J}{\partial b}$. Đặt **phần dư (residual)**
$r_i = \hat{y}_i - y_i$. Áp **quy tắc chuỗi** (Chương 3, mục 2.8): "đạo hàm của bình phương" nhân "đạo
hàm của bên trong":

$$ \frac{\partial}{\partial w}(\hat{y}_i - y_i)^2 = 2(\hat{y}_i - y_i)\cdot\frac{\partial \hat{y}_i}{\partial w}
   = 2 r_i\, x_i, \qquad \frac{\partial \hat{y}_i}{\partial w} = x_i. $$

Lấy trung bình toàn bộ điểm:

$$ \boxed{\;\frac{\partial J}{\partial w} = \frac{2}{N}\sum_{i=1}^{N} r_i\, x_i, \qquad
   \frac{\partial J}{\partial b} = \frac{2}{N}\sum_{i=1}^{N} r_i\;} $$

(với $b$: $\partial \hat{y}_i/\partial b = 1$). Dạng vector cho nhiều đặc trưng:
$\nabla_{\mathbf{w}} J = \frac{2}{N} X^\top \mathbf{r}$ và $\partial J/\partial b = \frac{2}{N}\sum r_i$.

**Ví dụ tính tay (khớp với output ở mục 4).** Dữ liệu đã chuẩn hóa
$x = [-1.3416, -0.4472, 0.4472, 1.3416]$, $y = [2, 4, 6, 8]$, khởi tạo $w = b = 0$. Khi đó $\hat{y} = 0$
nên $r = \hat{y} - y = [-2, -4, -6, -8]$:

$$ \frac{\partial J}{\partial w} = \frac{2}{4}\big[(-2)(-1.3416) + (-4)(-0.4472) + (-6)(0.4472) + (-8)(1.3416)\big]
   = \tfrac{1}{2}(-8.9443) = -4.4721, $$
$$ \frac{\partial J}{\partial b} = \frac{2}{4}\big[(-2) + (-4) + (-6) + (-8)\big] = \tfrac{1}{2}(-20) = -10. $$

### 2.4. Cập nhật & chuẩn hóa

**Gradient descent** (Chương 3, mục 2.6), với tốc độ học $\eta$:

$$ w \leftarrow w - \eta\,\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial J}{\partial b}. $$

Với ví dụ trên, $\eta = 0.1$: $w \leftarrow 0 - 0.1\cdot(-4.4721) = 0.4472$ và
$b \leftarrow 0 - 0.1\cdot(-10) = 1.0$. Sau đó lặp lại — mỗi vòng đường thẳng khít hơn một chút.

**Chuẩn hóa (standardization)** dùng công thức: với mỗi đặc trưng,

$$ x' = \frac{x - \mu}{\sigma}, $$

trong đó $\mu$ là trung bình và $\sigma$ là độ lệch chuẩn **tính từ dữ liệu huấn luyện**. Điểm mấu chốt:
khi gặp dữ liệu mới, ta dùng **đúng** $\mu, \sigma$ đã học — không tính lại — để mọi thứ nhất quán (lớp
`Standardizer` lo việc này qua `fit` rồi `transform`).

---

## 3. Cài đặt (Implementation)

### 3.1. Công cụ chuẩn hóa — `ddl/preprocessing.py`

Đây là module dùng chung đầu tiên có "ruột". Lớp `Standardizer` ghi nhớ $\mu, \sigma$ lúc `fit`, rồi áp
lại lúc `transform`:

```python
# ddl/preprocessing.py
import numpy as np

class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)                       # std chia cho N
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)  # tránh chia 0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_             # x' = (x - mean) / std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std_ + self.mean_
```

### 3.2. Huấn luyện — `ch04_linear_regression.py`

Toàn bộ vòng đời thu nhỏ: chuẩn hóa → khởi tạo → lặp (forward → loss → gradient → update) → suy luận.

```python
import numpy as np
from ddl.preprocessing import Standardizer

np.random.seed(42)

# (1) Dữ liệu tuyến tính hoàn hảo y = 2x (để dễ kiểm chứng)
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])

# (2) Chuẩn hóa đặc trưng
scaler = Standardizer()
x_std = scaler.fit_transform(x)

# (3) Khởi tạo tham số
w, b = 0.0, 0.0
lr, n_steps = 0.1, 50

# (4) Vòng lặp huấn luyện
for step in range(n_steps + 1):
    y_pred = w * x_std + b                  # forward
    loss = np.mean((y_pred - y) ** 2)       # MSE
    residual = y_pred - y                   # r = y_hat - y
    grad_w = 2 * np.mean(residual * x_std)  # dJ/dw = (2/N) sum(r*x)
    grad_b = 2 * np.mean(residual)          # dJ/db = (2/N) sum(r)
    w -= lr * grad_w                        # cập nhật
    b -= lr * grad_b

# (6) Suy luận trên x mới (dùng transform, KHÔNG fit lại)
x_new_std = scaler.transform(np.array([5.0]))
y_new = w * x_new_std + b
```

(Bản đầy đủ trong `ch04_linear_regression.py` còn in tiến trình mỗi 10 bước.)

---

## 4. Chạy thử (Run it)

```powershell
python ch04_linear_regression.py
```

**Kết quả mong đợi (expected output)** — vì dữ liệu là tuyến tính hoàn hảo, mất mát giảm rất nhanh về 0
và $(w, b)$ tiến về nghiệm đúng. Vài chữ số cuối có thể hơi khác trên máy bạn:

```
x sau chuẩn hóa: [-1.3416 -0.4472  0.4472  1.3416]
step   0 | loss = 30.000000 | w = 0.0000, b = 0.0000
step  10 | loss = 0.345876 | w = 1.9960, b = 4.4631
step  20 | loss = 0.003988 | w = 2.2103, b = 4.9424
step  30 | loss = 0.000046 | w = 2.2333, b = 4.9938
step  40 | loss = 0.000001 | w = 2.2358, b = 4.9993
step  50 | loss = 0.000000 | w = 2.2360, b = 4.9999
Học xong: w = 2.2360, b = 4.9999
Dự đoán cho x=5: 9.9999 (đúng phải là 10)
```

Đọc kết quả này thế nào?

- **Mất mát đi từ 30 về ~0:** mô hình thực sự **đang học**. Đúng tinh thần "lặp lại để giảm loss".
- **$b \to 5$** chính là **trung bình của $y$** ($\frac{2+4+6+8}{4} = 5$): vì $x$ đã chuẩn hóa về trung
  bình 0, nên $b$ học đúng "giá trị trung bình". **$w \to 2.236$** là độ dốc *trong không gian đã chuẩn
  hóa*.
- **Suy luận:** với $x = 5$, mô hình đoán $\approx 10$ — đúng quan hệ $y = 2x$, dù lúc huấn luyện nó chưa
  hề thấy $x = 5$. Đây là cái đẹp của học máy: học được **quy luật**, không chỉ học thuộc.

> Vì sao $w$ và $b$ học "gọn" như vậy? Nhờ chuẩn hóa, bài toán tách đôi: cập nhật $w$ không vướng $b$ và
> ngược lại, nên mỗi tham số trượt thẳng về đáy theo cấp số nhân. Đó chính là lợi ích "cái bát tròn" ở
> mục 1.4, thấy được bằng số.

---

## 5. Bài tập (Exercises)

1. **Đổi tốc độ học.** Chạy lại với `lr = 0.5`, rồi `lr = 1.0`, rồi `lr = 1.1`. Mô tả điều gì xảy ra với
   mất mát ở mỗi mức. (Gợi ý: để ý hệ số $1 - 2\eta$ trong cập nhật.)
2. **Thêm nhiễu.** Đổi `y` thành `y = 2 * x + np.random.randn(N) * 0.5` (thêm **nhiễu — noise**). Mất mát
   cuối còn về đúng 0 không? $w, b$ học được có còn gần nghiệm cũ không?
3. **Vì sao `transform` chứ không `fit_transform` cho dữ liệu mới?** Giải thích bằng lời chuyện gì sẽ sai
   nếu ở bước (6) ta gọi `scaler.fit_transform(x_new)` thay vì `scaler.transform(x_new)`.
4. **Gradient cho một điểm.** Với một điểm duy nhất $x = 2, y = 5$ và $w = 1, b = 0$: tính $\hat{y}$, phần
   dư $r$, rồi $\partial J/\partial w$ và $\partial J/\partial b$ (dùng $N = 1$).
5. **Bỏ chuẩn hóa.** Huấn luyện trực tiếp trên `x` gốc (không chuẩn hóa) với cùng `lr = 0.1`. Mất mát có
   còn giảm êm không, hay bị "nổ" (tăng vọt)? Vì sao?

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. $\eta = 0.5$: hệ số $1 - 2\eta = 0$, hội tụ **chỉ trong 1 bước**. $\eta = 1.0$: hệ số $= -1$, $w$ và
   $b$ **dao động** quanh nghiệm mãi không hội tụ. $\eta = 1.1$: hệ số $-1.2$ (độ lớn $> 1$), mất mát
   **phát tán** (tăng vọt). Đây là minh họa sống động cho "lr quá lớn" ở Chương 3.
2. Mất mát cuối **không** về đúng 0 (vì dữ liệu giờ không nằm trọn trên một đường thẳng), nhưng $w, b$
   vẫn **gần** nghiệm cũ — mô hình tìm đường thẳng *khít nhất có thể*.
3. Nếu `fit_transform(x_new)`, ta sẽ tính lại $\mu, \sigma$ từ *riêng* dữ liệu mới → một thang đo khác →
   mô hình diễn giải sai đầu vào. Phải dùng **đúng** $\mu, \sigma$ đã học từ train.
4. $\hat{y} = 1\cdot 2 + 0 = 2$; $r = 2 - 5 = -3$; $\partial J/\partial w = 2 r x = 2(-3)(2) = -12$;
   $\partial J/\partial b = 2 r = -6$.
5. Không chuẩn hóa, $x$ có thang đo lớn hơn → gradient lớn → với `lr = 0.1` rất dễ **bước quá đà** khiến
   mất mát tăng vọt (phát tán). Đây đúng là vấn đề "thung lũng méo" mà chuẩn hóa giải quyết.

</details>

---

## 6. Tiếp theo (What's next)

Bạn vừa xây mô hình biết tự học đầu tiên cho bài toán **dự đoán số**. Ở **Chương 5 — Phân loại nhị phân
từ số 0 (Logistic regression from scratch)**, ta chuyển sang bài toán **phân loại (classification)**: thay
vì đoán một con số, mô hình sẽ đoán **xác suất thuộc về một lớp**. Ta sẽ gặp hai người bạn mới — hàm
**sigmoid** và hàm mất mát **cross-entropy** — nhưng bộ khung "mô hình → mất mát → gradient → cập nhật"
thì vẫn y nguyên.
