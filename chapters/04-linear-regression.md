# Chương 4 — Hồi quy tuyến tính từ số 0 (Linear regression from scratch)

## 0. Chúng ta đang ở đâu (Where we are)

- **Chương 1:** bản đồ quy trình (pipeline).
- **Chương 2:** công cụ NumPy.
- **Chương 3:** đạo hàm, gradient, **gradient descent**, quy tắc chuỗi.

Đây là chương **bước ngoặt**: lần đầu tiên ta ghép tất cả lại thành một **mô hình biết tự học**. Trên
bản đồ pipeline, chương này đi gần trọn một vòng trên một bài toán tí hon:

> (4) **xây mô hình** (đường thẳng) → (5) **hàm mất mát** (MSE) → (6) **vòng lặp huấn luyện** (gradient
> descent) → (9) **suy luận** (dự đoán cho ngày mới).

(Hai ô (2) **chia dữ liệu** và (3) **tiền xử lý** ta để dành đến Chương 8 — ở đây dữ liệu chỉ vài điểm
minh họa nên chưa cần. Cứ giữ pipeline đầy đủ trong đầu; ta sẽ lấp nốt sau.)

Mọi thứ viết tay bằng NumPy — không thư viện deep learning nào — để bạn thấy rõ từng bánh răng.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Bài toán: dự đoán một con số (regression)

**Hồi quy (regression)** nghĩa là dự đoán một **con số liên tục**: giá nhà, nhiệt độ, doanh thu… (Khác
với **phân loại — classification** ở Chương 5, vốn dự đoán một *nhãn*: chó/mèo, số 0–9.)

Ví dụ xuyên suốt chương: một xe bán nước ghi lại 5 ngày — **nhiệt độ** (so với mốc 25°C) và **số ly bán
được**:

| Nhiệt độ (so với 25°C) — $x$ | -2 | -1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| Số ly bán — $y$ | 12 | 16 | 20 | 24 | 28 |

(Vì sao ghi nhiệt độ dưới dạng "chênh lệch so với 25°C" mà không phải nhiệt độ thật? Chỉ để các con số
nhỏ và nằm quanh 0 — giúp việc học ổn định hơn. Cách *tự động* làm điều này cho dữ liệu bất kỳ gọi là
**chuẩn hóa — standardization**, ta sẽ học ở Chương 8.)

### 1.2. Mô hình: một đường thẳng

Giả thuyết đơn giản nhất: số ly tỉ lệ tuyến tính với nhiệt độ. Tức mô hình là một **đường thẳng**:

$$ \hat{y} = w\,x + b. $$

- $w$ — **trọng số (weight)** — là **độ dốc (slope)**: $x$ tăng 1 thì $\hat{y}$ tăng $w$.
- $b$ — **độ chệch (bias)** — là giá trị $\hat{y}$ khi $x = 0$ (giao điểm với trục tung).

$w$ và $b$ là **tham số (parameters)** mà mô hình phải *học*. Học xong = tìm được đường thẳng đi "vừa
khít" qua đám điểm.

### 1.3. Đo độ sai: hàm mất mát MSE

Làm sao biết một đường thẳng "khít" hay không? Phải **đo** độ sai. Với mỗi điểm, sai số là khoảng cách
giữa dự đoán $\hat{y}_i$ và giá trị thật $y_i$. Ta dùng **sai số toàn phương trung bình (Mean Squared
Error — MSE)**:

$$ J(w, b) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2. $$

Vì sao **bình phương**? (1) để sai số âm và dương không triệt tiêu nhau; (2) để **phạt nặng** những sai
lớn; (3) nó **trơn (smooth)**, dễ lấy đạo hàm — điều kiện sống còn để dùng gradient descent.

### 1.4. ⭐ Điểm cốt lõi cần khắc sâu: vì sao "học" thay vì "đoán"?

> Với 5 điểm trên, bạn nhìn cái là **đoán** ngay được: mỗi độ tăng thì bán thêm 4 ly ($w = 4$), và ở
> mốc 25°C ($x = 0$) bán được 20 ly ($b = 20$). Vậy cần học máy làm gì?
>
> **Vì mục tiêu không phải giải bài toán tí hon này.** Mục tiêu là *học thuộc QUY TRÌNH*:
> **mô hình → mất mát → gradient → cập nhật**. Quy trình ấy **y hệt nhau** dù dữ liệu là 5 điểm hay 5
> triệu điểm, 1 đặc trưng hay 1000 đặc trưng. Trên dữ liệu thật (giá nhà theo 50 yếu tố, ảnh 784 điểm
> ảnh…), **không ai "nhìn ra" được $w, b$** — nhưng gradient descent thì vẫn tự tìm ra. Ví dụ tí hon ở
> đây chỉ để bạn **thấy rõ từng bánh răng**; các chương sau ta giữ nguyên quy trình và chỉ phóng to quy
> mô. Hãy để ý: bên dưới, **ta không hề nói cho mô hình biết $w=4, b=20$** — nó phải tự mò ra từ con số
> 0.

### 1.5. Kế hoạch

Khớp đúng Chương 3: coi $J(w, b)$ là hàm cần **tối thiểu hóa**, tính **gradient** của nó theo $w, b$,
rồi **gradient descent** để mô hình tự trượt xuống đáy "cái bát" mất mát. (Ta sẽ *vẽ* cái bát đó ở mục 4
để thấy tận mắt.)

---

## 2. Toán học (The math)

### 2.1. Mô hình (một và nhiều đặc trưng)

Một đặc trưng: $\hat{y} = w x + b$. Nhiều đặc trưng ($D$ đặc trưng gói trong vector $\mathbf{x}$):

$$ \hat{y} = \mathbf{w}^\top \mathbf{x} + b, \qquad \text{cho cả lô: } \hat{\mathbf{y}} = X\mathbf{w} + b, $$

với $X$ shape $(N, D)$, $\mathbf{w}$ shape $(D,)$. (Chương này dùng $D = 1$ cho thật rõ; dạng vector sẽ
trở lại ở Chương 6.)

### 2.2. Hàm mất mát

$$ J(w, b) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2, \qquad \hat{y}_i = w x_i + b. $$

### 2.3. Tính gradient bằng quy tắc chuỗi

Ta cần $\dfrac{\partial J}{\partial w}$ và $\dfrac{\partial J}{\partial b}$. Đặt **phần dư (residual)**
$r_i = \hat{y}_i - y_i$. Áp **quy tắc chuỗi** (Chương 3, mục 2.8) — "đạo hàm của bình phương" nhân "đạo
hàm của bên trong":

$$ \frac{\partial}{\partial w}(\hat{y}_i - y_i)^2 = 2(\hat{y}_i - y_i)\cdot\frac{\partial \hat{y}_i}{\partial w}
   = 2 r_i\, x_i \qquad (\text{vì } \tfrac{\partial \hat{y}_i}{\partial w} = x_i). $$

Lấy trung bình toàn bộ điểm:

$$ \boxed{\;\frac{\partial J}{\partial w} = \frac{2}{N}\sum_{i=1}^{N} r_i\, x_i, \qquad
   \frac{\partial J}{\partial b} = \frac{2}{N}\sum_{i=1}^{N} r_i\;} $$

(với $b$: $\partial \hat{y}_i/\partial b = 1$). Dạng vector cho nhiều đặc trưng:
$\nabla_{\mathbf{w}} J = \frac{2}{N} X^\top \mathbf{r}$ và $\partial J/\partial b = \frac{2}{N}\sum_i r_i$.

**Ví dụ tính tay (khớp với output ở mục 4).** Lúc bắt đầu $w = b = 0$, nên $\hat{y} = 0$ và
$r = \hat{y} - y = [-12, -16, -20, -24, -28]$. Với $x = [-2, -1, 0, 1, 2]$:

$$ \frac{\partial J}{\partial w} = \frac{2}{5}\big[(-12)(-2) + (-16)(-1) + (-20)(0) + (-24)(1) + (-28)(2)\big]
   = \frac{2}{5}(-40) = -16, $$
$$ \frac{\partial J}{\partial b} = \frac{2}{5}\big[(-12) + (-16) + (-20) + (-24) + (-28)\big] = \frac{2}{5}(-100) = -40. $$

### 2.4. Cập nhật (gradient descent)

Với tốc độ học $\eta$ (Chương 3, mục 2.6):

$$ w \leftarrow w - \eta\,\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial J}{\partial b}. $$

Dùng ví dụ trên, $\eta = 0.1$: $w \leftarrow 0 - 0.1\cdot(-16) = 1.6$ và $b \leftarrow 0 - 0.1\cdot(-40)
= 4.0$. Gradient **âm** nên tham số **tăng lên** — đi đúng hướng về phía nghiệm $w=4, b=20$. Lặp lại
nhiều vòng, đường thẳng khít dần.

---

## 3. Cài đặt (Implementation)

### 3.1. Huấn luyện — `ch04_linear_regression.py`

Toàn bộ vòng đời thu nhỏ: khởi tạo → lặp (forward → loss → gradient → update) → suy luận. Không thư viện
deep learning, không tiền xử lý — chỉ NumPy thuần.

```python
import numpy as np

# (1) Dữ liệu 5 ngày: x = nhiệt độ lệch so với 25°C, y = số ly bán được
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = np.array([12.0, 16.0, 20.0, 24.0, 28.0])

# (2) Khởi tạo tham số từ 0 — mô hình CHƯA biết gì
w, b = 0.0, 0.0
lr, n_steps = 0.1, 50

# (3) Vòng lặp huấn luyện
for step in range(n_steps + 1):
    y_pred = w * x + b                       # forward
    loss = np.mean((y_pred - y) ** 2)        # MSE
    residual = y_pred - y                    # r = y_hat - y
    grad_w = 2 * np.mean(residual * x)       # dJ/dw = (2/N) sum(r*x)
    grad_b = 2 * np.mean(residual)           # dJ/db = (2/N) sum(r)
    w -= lr * grad_w                         # cập nhật theta <- theta - lr*grad
    b -= lr * grad_b

# (5) Suy luận cho ngày mới 26.5°C (x = 1.5)
y_new = w * np.array([1.5]) + b
```

(Bản đầy đủ trong `ch04_linear_regression.py` còn in tiến trình mỗi 10 bước.)

### 3.2. Vẽ "cái bát" mất mát — `ch04_plot_loss.py`

Để *thấy* gradient descent đang làm gì, ta vẽ loss như một hàm của $w$ (cố định $b$ ở giá trị tốt nhất
để được đường cong 1 chiều). Nó có hình **cái bát (parabola)**, và các bước gradient descent là những
điểm **lăn dần xuống đáy**:

```
 loss
  |  *                                   *
  |    *                               *
  |      *                           *
  |        *                       *
  |          ●  (bắt đầu, w=0)   *
  |            ●               *
  |              ●           *
  |                ●  ●  ●  ●            <- lăn xuống đáy
  +------------------+------------------- w
                   w*=4  (đáy: loss nhỏ nhất)
```

Đoạn code (rút gọn) tạo đúng hình trên rồi lưu ra `ch04_loss.png`:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = np.array([12.0, 16.0, 20.0, 24.0, 28.0])
b = 20.0                                    # cố định b để vẽ loss theo riêng w

def loss_for_w(w):
    return np.mean((w * x + b - y) ** 2)

w_grid = np.linspace(-2, 10, 100)           # nhiều giá trị w để vẽ "cái bát"
loss_grid = [loss_for_w(w) for w in w_grid]

w, lr, w_steps, loss_steps = 0.0, 0.1, [], []
for _ in range(8):                          # 8 bước gradient descent
    w_steps.append(w); loss_steps.append(loss_for_w(w))
    w -= lr * 2 * np.mean((w * x + b - y) * x)

plt.plot(w_grid, loss_grid)                 # cái bát
plt.scatter(w_steps, loss_steps, color="red")   # các bước lăn xuống
plt.savefig("ch04_loss.png")
```

---

## 4. Chạy thử (Run it)

**(a) Huấn luyện:**

```powershell
python ch04_linear_regression.py
```

**Kết quả mong đợi (expected output)** — dữ liệu nằm trọn trên một đường thẳng nên mất mát giảm rất
nhanh về 0, và $(w, b)$ tiến về đúng $(4, 20)$. Vài chữ số cuối có thể hơi khác trên máy bạn:

```
step   0 | loss = 432.000000 | w = 0.0000, b = 0.0000
step  10 | loss = 4.612856 | w = 3.9758, b = 17.8525
step  20 | loss = 0.053169 | w = 3.9999, b = 19.7694
step  30 | loss = 0.000613 | w = 4.0000, b = 19.9752
step  40 | loss = 0.000007 | w = 4.0000, b = 19.9973
step  50 | loss = 0.000000 | w = 4.0000, b = 19.9997
Học xong: w = 4.0000, b = 19.9998  (quan hệ thật: w=4, b=20)
Dự đoán cho ngày 26.5°C (x=1.5): 25.9998 (đúng phải là 26)
```

**(b) Vẽ hàm mất mát:**

```powershell
python ch04_plot_loss.py
```

```
Đã lưu hình vào ch04_loss.png
```

Mở `ch04_loss.png`: bạn sẽ thấy một đường cong **hình chữ U** (cái bát), với các **chấm đỏ** bắt đầu ở
$w = 0$ (mất mát cao) rồi **lăn dần xuống đáy** ở $w = 4$ (mất mát thấp nhất). Đây chính là gradient
descent — đúng hình ảnh "quả bóng lăn xuống đáy" ở Chương 3, giờ thấy bằng mắt.

Đọc kết quả huấn luyện thế nào?

- **Mất mát đi từ 432 về ~0:** mô hình thực sự **đang học**, đúng tinh thần "lặp lại để giảm loss".
- **$w \to 4$ và $b \to 20$:** mô hình **tự mò ra** đúng quan hệ $y = 4x + 20$ — dù ta **chưa hề nói**
  cho nó biết. Nó chỉ có dữ liệu và quy tắc cập nhật.
- **Suy luận:** với ngày 26.5°C (một điểm **chưa từng thấy**), mô hình đoán $\approx 26$ ly — đúng quy
  luật. Đây là cái đẹp của học máy: học được **quy luật**, không chỉ học thuộc.

---

## 5. Bài tập (Exercises)

1. **Đổi tốc độ học.** Chạy lại với `lr = 0.4`, rồi `lr = 0.5`, rồi `lr = 0.6`. Mô tả điều gì xảy ra với
   mất mát ở mỗi mức (hội tụ nhanh hơn? dao động? phát tán?).
2. **Thêm nhiễu (noise).** Đổi `y` thành `y = 4 * x + 20 + np.random.randn(N) * 1.0`. Mất mát cuối còn về
   đúng 0 không? $w, b$ học được có còn gần $(4, 20)$ không? Vì sao?
3. **Tự đoán vs để máy học.** Viết 2–3 câu giải thích bằng lời: nếu ta đã đoán được $w=4, b=20$ bằng
   mắt, thì *giá trị thật sự* của việc cho mô hình tự học nằm ở đâu? (Liên hệ mục 1.4.)
4. **Gradient cho một điểm.** Với một điểm duy nhất $x = 2, y = 30$ và $w = 1, b = 0$: tính $\hat{y}$,
   phần dư $r$, rồi $\partial J/\partial w$ và $\partial J/\partial b$ (dùng $N = 1$).
5. **Đọc cái bát.** Trong hình `ch04_loss.png`, vì sao các chấm đỏ ở gần đáy lại **sát nhau hơn** so với
   các chấm ở trên cao? (Gợi ý: độ dốc tại đó thế nào, nên bước đi dài hay ngắn?)

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. `lr = 0.4`: hội tụ nhanh hơn. `lr = 0.5`: với phần $w$ (hệ số $1 - 4\eta = -1$ tại $\eta=0.5$) bắt đầu
   **dao động**. `lr = 0.6`: vượt ngưỡng ổn định → mất mát **phát tán** (tăng vọt). Đúng minh họa "lr quá
   lớn" ở Chương 3.
2. Mất mát cuối **không** về đúng 0 (dữ liệu giờ không nằm trọn trên một đường thẳng vì có nhiễu), nhưng
   $w, b$ vẫn **gần** $(4, 20)$ — mô hình tìm đường thẳng *khít nhất có thể*. Đây cũng là tình huống của
   dữ liệu thật.
3. Giá trị thật nằm ở **quy trình** (model → loss → gradient → update): nó hoạt động *y hệt* trên các bài
   toán mà ta **không thể** đoán bằng mắt (nhiều điểm, nhiều đặc trưng). Ví dụ tí hon chỉ để kiểm chứng
   quy trình đúng.
4. $\hat{y} = 1\cdot 2 + 0 = 2$; $r = 2 - 30 = -28$; $\partial J/\partial w = 2 r x = 2(-28)(2) = -112$;
   $\partial J/\partial b = 2 r = -56$.
5. Càng gần đáy, độ dốc (gradient) càng **nhỏ**, nên mỗi bước $-\eta\,\text{grad}$ càng **ngắn** → các
   chấm sát nhau hơn. Đáy là nơi gradient bằng 0, mô hình gần như dừng lại.

</details>

---

## 6. Tiếp theo (What's next)

Bạn vừa xây mô hình biết tự học đầu tiên cho bài toán **dự đoán số**. Ở **Chương 5 — Phân loại nhị phân
từ số 0 (Logistic regression from scratch)**, ta chuyển sang bài toán **phân loại (classification)**:
thay vì đoán một con số, mô hình sẽ đoán **xác suất thuộc về một lớp**. Ta gặp hai người bạn mới — hàm
**sigmoid** và hàm mất mát **cross-entropy** — nhưng bộ khung "mô hình → mất mát → gradient → cập nhật"
thì **vẫn y nguyên**.
