# Chương 5 — Phân loại nhị phân từ số 0 (Logistic regression from scratch)

## 0. Chúng ta đang ở đâu (Where we are)

Ở **Chương 4** ta xây mô hình đầu tiên cho bài toán **dự đoán số (regression)** với khung bốn bước:
**mô hình → mất mát → gradient → cập nhật**. Chương này giải một bài toán *khác loại* — **phân loại
(classification)** — nhưng dùng **đúng khung đó**. Điểm thú vị: ta chỉ thay **hai mảnh** (hàm
**sigmoid** và hàm mất mát **cross-entropy**), còn vòng lặp huấn luyện thì gần như sao chép từ Chương 4.

Trên bản đồ pipeline, ta vẫn ở các ô (4) **mô hình** → (5) **mất mát** → (6) **vòng lặp huấn luyện** →
(9) **suy luận**, vẫn viết tay bằng NumPy.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Phân loại khác hồi quy ở chỗ nào?

**Phân loại (classification)** dự đoán một **nhãn (label)**, không phải một con số liên tục. **Phân loại
nhị phân (binary classification)** là khi chỉ có **hai** nhãn: đậu/rớt, spam/không spam, có bệnh/khỏe.
Ta quy ước hai nhãn là **0** và **1**.

Ví dụ xuyên suốt chương: dự đoán một sinh viên **đậu (1)** hay **rớt (0)** dựa trên **số giờ ôn thi**.

| Giờ ôn (so với mốc 3.5h) — $x$ | -2.5 | -1.5 | -0.5 | 0.5 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|
| Kết quả — $y$ | 0 | 0 | 0 | 1 | 1 | 1 |

### 1.2. Vì sao không dùng thẳng đường thẳng như Chương 4?

Đường thẳng $wx + b$ cho ra **mọi số thực** từ $-\infty$ đến $+\infty$. Nhưng ta muốn một **xác suất
(probability)** — một số trong khoảng $[0, 1]$ trả lời "khả năng đậu là bao nhiêu?". Cần một cách **ép
(squash)** đầu ra của đường thẳng về khoảng $(0, 1)$. Người làm việc đó là hàm **sigmoid**:

$$ \sigma(z) = \frac{1}{1 + e^{-z}}. $$

Hình chữ **S** của nó: số rất âm → gần 0, số 0 → đúng 0.5, số rất dương → gần 1:

```
 p
1.0|                         _____------
   |                  ___----
0.5|---------------●------------------   <- z=0 cho p=0.5
   |        ___----
0.0|___-----
   +-------------------------------------- z
        âm          0          dương
```

### 1.3. Mô hình logistic

Ghép lại: vẫn tính phần tuyến tính $z = wx + b$ y như Chương 4, rồi đưa qua sigmoid để ra **xác suất**:

$$ p = \sigma(wx + b) = P(y = 1 \mid x). $$

**Quyết định (decision):** nếu $p \ge 0.5$ thì đoán nhãn **1**, ngược lại đoán **0**. Vì $\sigma(z) \ge
0.5 \Leftrightarrow z \ge 0$, **ranh giới quyết định (decision boundary)** nằm đúng tại $z = 0$, tức
$wx + b = 0$. Trong ví dụ của ta, nó rơi vào $x = 0$ (học 3.5 giờ).

### 1.4. Vì sao cần hàm mất mát mới (cross-entropy)?

Ta cần đo "dự đoán xác suất sai bao nhiêu". MSE của Chương 4 dùng được nhưng kém: ghép với sigmoid nó
tạo ra mặt mất mát **lồi lõm khó tối ưu** và gradient hay bị **triệt tiêu (vanish)**. Thay vào đó ta
dùng **cross-entropy nhị phân (Binary Cross-Entropy — BCE)**:

$$ J = -\frac{1}{N}\sum_{i=1}^{N}\Big[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\Big]. $$

**Trực giác:** với một điểm có nhãn thật $y = 1$, phần đóng góp là $-\log p$ — muốn nhỏ thì $p$ phải
**gần 1**. Với $y = 0$, phần đóng góp là $-\log(1 - p)$ — muốn nhỏ thì $p$ phải **gần 0**. Và vì
$-\log(\text{số gần 0}) \to +\infty$, cross-entropy **phạt cực nặng** kiểu dự đoán **tự tin nhưng sai**
(ví dụ nói chắc chắn đậu, $p = 0.99$, mà thực ra rớt).

---

## 2. Toán học (The math)

### 2.1. Sigmoid và đạo hàm tuyệt đẹp của nó

$$ \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \boxed{\;\sigma'(z) = \sigma(z)\,(1 - \sigma(z))\;}. $$

Đạo hàm này gọn đến bất ngờ — đầu ra của sigmoid tự quyết định độ dốc của chính nó. (Ta sẽ gặp lại nó ở
**lan truyền ngược** Chương 7.) Kiểm chứng nhanh tại $z = 0$: $\sigma = 0.5$ nên $\sigma' = 0.5\cdot 0.5
= 0.25$ — đúng là chỗ dốc nhất của hình chữ S.

### 2.2. Mô hình & mất mát

$$ p_i = \sigma(z_i), \quad z_i = w x_i + b, \qquad
   J = -\frac{1}{N}\sum_i \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]. $$

### 2.3. Gradient — và một bất ngờ đẹp đẽ

Trông $J$ có vẻ đáng sợ, nhưng khi áp **quy tắc chuỗi** (Chương 3), mọi thứ rút gọn kỳ diệu. Xét một
điểm, đi từ $J$ ngược về $z$ qua ba mắt xích $J \leftarrow p \leftarrow z$:

$$ \frac{\partial J}{\partial p} = -\Big(\frac{y}{p} - \frac{1 - y}{1 - p}\Big), \qquad
   \frac{\partial p}{\partial z} = \sigma'(z) = p(1 - p). $$

Nhân hai mắt xích lại, $p(1-p)$ **triệt tiêu** đúng các mẫu số:

$$ \frac{\partial J}{\partial z} = -\Big(\frac{y}{p} - \frac{1-y}{1-p}\Big)\,p(1-p)
   = -\big[\,y(1-p) - (1-y)p\,\big] = -\,(y - p) = p - y. $$

Vậy $\dfrac{\partial J}{\partial z} = p - y$ — **chính là phần dư (residual)**, y như Chương 4! Thêm
$\partial z/\partial w = x$ và $\partial z/\partial b = 1$:

$$ \boxed{\;\frac{\partial J}{\partial w} = \frac{1}{N}\sum_i (p_i - y_i)\,x_i, \qquad
   \frac{\partial J}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i)\;} $$

Giống hệt công thức Chương 4, chỉ khác: $\hat{y}$ (dự đoán số) được thay bằng $p$ (xác suất), và **không
còn hệ số 2** (vì cross-entropy không có bình phương; sự triệt tiêu của $\sigma'$ lo phần còn lại). Đây
không phải trùng hợp — cặp sigmoid + cross-entropy được *thiết kế* để gradient ra gọn như vậy.

**Ví dụ tính tay (khớp output mục 4).** Lúc đầu $w = b = 0$ nên mọi $z = 0$, mọi $p = 0.5$. Phần dư
$r = p - y = [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]$. Với $x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]$:

$$ \frac{\partial J}{\partial w} = \frac{1}{6}\big[(0.5)(-2.5) + (0.5)(-1.5) + (0.5)(-0.5) + (-0.5)(0.5)
   + (-0.5)(1.5) + (-0.5)(2.5)\big] = \frac{-4.5}{6} = -0.75, $$

còn $\dfrac{\partial J}{\partial b} = \frac{1}{6}(0.5 + 0.5 + 0.5 - 0.5 - 0.5 - 0.5) = 0$. Vì dữ liệu
**đối xứng**, $b$ sẽ đứng yên ở 0 suốt quá trình; chỉ $w$ thay đổi. Với $\eta = 0.5$:
$w \leftarrow 0 - 0.5\cdot(-0.75) = 0.375$.

### 2.4. Cập nhật

Vẫn là gradient descent quen thuộc:

$$ w \leftarrow w - \eta\,\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\,\frac{\partial J}{\partial b}. $$

---

## 3. Cài đặt (Implementation)

### 3.1. Các hàm dùng chung — `ddl/functions.py`

Ta gom các hàm hay dùng vào một module. Chương này cần `sigmoid` và `bce_loss`; `softmax`, `relu`,
`mse_loss` để dành cho các chương sau (đã đặt sẵn theo Hợp đồng khóa học):

```python
# ddl/functions.py
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))            # ép z về (0, 1)

def bce_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)             # tránh log(0)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
```

### 3.2. Huấn luyện — `ch05_logistic_regression.py`

So với `ch04`, chỉ khác đúng **hai dòng** (thêm `sigmoid`, đổi loss sang `bce_loss`):

```python
import numpy as np
from ddl.functions import sigmoid, bce_loss

x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
w, b = 0.0, 0.0
lr, n_steps = 0.5, 8

for step in range(n_steps + 1):
    z = w * x + b                  # phần tuyến tính
    p = sigmoid(z)                 # <-- MỚI: ép thành xác suất
    loss = bce_loss(p, y)          # <-- MỚI: cross-entropy thay cho MSE
    residual = p - y               # r = p - y  (vẫn là "phần dư")
    grad_w = np.mean(residual * x) # dJ/dw  (KHÔNG còn hệ số 2)
    grad_b = np.mean(residual)     # dJ/db
    w -= lr * grad_w
    b -= lr * grad_b
```

(Bản đầy đủ trong `ch05_logistic_regression.py` còn in độ chính xác và xác suất từng điểm.)

### 3.3. Vẽ đường cong đã học — `ch05_plot_sigmoid.py`

Để *thấy* mô hình phân chia hai lớp, ta vẽ đường cong $p = \sigma(wx + b)$ cùng 6 điểm dữ liệu và ranh
giới quyết định tại $x = 0$ (file `ch05_plot_sigmoid.py`, lưu ra `ch05_sigmoid.png`).

---

## 4. Chạy thử (Run it)

**(a) Huấn luyện:**

```powershell
python ch05_logistic_regression.py
```

**Kết quả mong đợi (expected output)** — các giá trị `loss`, `w`, và xác suất là **xấp xỉ** (sigmoid
sinh số thập phân lẻ); điều cần thấy là **loss giảm đều**, **acc đạt 1.000**, và **xác suất tách dần về
hai phía 0 và 1**:

```
step 0 | loss = 0.6931 | w = 0.0000, b = 0.0000 | acc = 0.500
step 1 | loss = 0.4617 | w = 0.3750, b = 0.0000 | acc = 1.000
step 2 | loss = 0.3581 | w = 0.6208, b = 0.0000 | acc = 1.000
step 3 | loss = 0.3012 | w = 0.7995, b = 0.0000 | acc = 1.000
step 4 | loss = 0.2648 | w = 0.9406, b = 0.0000 | acc = 1.000
step 5 | loss = 0.2394 | w = 1.0579, b = 0.0000 | acc = 1.000
step 6 | loss = 0.2202 | w = 1.1589, b = 0.0000 | acc = 1.000
step 7 | loss = 0.2051 | w = 1.2480, b = 0.0000 | acc = 1.000
step 8 | loss = 0.1929 | w = 1.3280, b = 0.0000 | acc = 1.000
Học xong: w = 1.4009, b = 0.0000
Độ chính xác trên tập huấn luyện: 1.000
Xác suất dự đoán cho từng điểm:
  x=-2.5, y=0 -> p(đậu)=0.0292
  x=-1.5, y=0 -> p(đậu)=0.1090
  x=-0.5, y=0 -> p(đậu)=0.3317
  x=+0.5, y=1 -> p(đậu)=0.6683
  x=+1.5, y=1 -> p(đậu)=0.8910
  x=+2.5, y=1 -> p(đậu)=0.9707
Sinh viên học 3.7 giờ (x=0.2): p(đậu) = 0.5696 -> dự đoán ĐẬU
```

**(b) Vẽ đường cong:**

```powershell
python ch05_plot_sigmoid.py
```

```
Đã lưu hình vào ch05_sigmoid.png
```

Đọc kết quả thế nào?

- **`loss` giảm từ 0.6931** (chính là $-\log 0.5$, mức "đoán mò 50–50") xuống dần: mô hình đang học.
- **`b` đứng yên ở 0** đúng như ta tiên đoán từ tính đối xứng của dữ liệu (mục 2.3).
- **Xác suất tách đôi:** các sinh viên rớt được gán $p$ thấp (0.03–0.33), các sinh viên đậu được gán $p$
  cao (0.67–0.97). Càng học nhiều bước, hai phía càng dạt xa về 0 và 1.
- **Suy luận có "độ tự tin":** sinh viên học 3.7 giờ (chỉ hơn mốc một chút) được đoán **đậu nhưng chỉ
  57%** — mô hình không chỉ nói đậu/rớt, nó nói **mức độ chắc chắn**. Đó là giá trị của việc xuất ra
  *xác suất* thay vì chỉ một nhãn cứng.

> Nhắc lại điểm cốt lõi (từ Chương 4): với 6 điểm này bạn cũng tự phân loại được bằng mắt. Quan trọng là
> **quy trình** — model → loss → gradient → update — chạy y hệt trên dữ liệu thật mà không ai phân loại
> thủ công nổi.

---

## 5. Bài tập (Exercises)

1. **Đạo hàm sigmoid.** Dùng $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, tính độ dốc của sigmoid tại
   $z = 0$ và tại $z = 2$ (cho biết $\sigma(2) \approx 0.88$). Ở đâu sigmoid dốc hơn?
2. **Đọc xác suất.** Mô hình học xong cho $p = 0.9$ với một điểm có nhãn thật $y = 1$, và $p = 0.2$ với
   một điểm $y = 0$. Tính phần đóng góp cross-entropy $-\log p$ và $-\log(1-p)$ cho mỗi điểm (dùng
   $\ln$). Điểm nào "bị phạt" nặng hơn?
3. **Học lâu hơn.** Tăng `n_steps` lên 50. Xác suất từng điểm thay đổi ra sao (gần 0/1 hơn không)? `loss`
   có về 0 tuyệt đối không? Vì sao với dữ liệu **tách rời được (separable)** thì $w$ cứ lớn dần mãi?
4. **Một điểm "khó".** Đổi nhãn của điểm $x = -0.5$ từ 0 thành 1 (giờ dữ liệu **không** tách rời hoàn
   toàn). Độ chính xác cuối còn 1.000 không? Mô hình xử lý điểm mâu thuẫn này thế nào (nhìn $p$ của nó)?
5. **Gradient tại một điểm.** Với $x = 1.0, y = 1$ và $w = 0.5, b = 0$: tính $z$, $p = \sigma(z)$ (cho
   $\sigma(0.5) \approx 0.622$), phần dư $r = p - y$, rồi $\partial J/\partial w$ (dùng $N = 1$).

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. Tại $z=0$: $\sigma' = 0.5\cdot 0.5 = 0.25$. Tại $z=2$: $\sigma' \approx 0.88\cdot 0.12 = 0.106$.
   Sigmoid **dốc hơn ở giữa** ($z=0$); ở hai đầu nó "phẳng" dần (đạo hàm nhỏ).
2. Điểm 1: $-\log 0.9 \approx 0.105$. Điểm 2: $-\log(1-0.2) = -\log 0.8 \approx 0.223$. Điểm 2 bị phạt
   nặng hơn (dự đoán kém tự tin hơn về phía đúng).
3. Xác suất tiến **sát 0 và 1** hơn; `loss` giảm nhưng **không** về đúng 0. Với dữ liệu tách rời được,
   tăng $w$ luôn làm các $p$ "chắc chắn" hơn nên loss còn giảm được mãi → $w$ lớn vô hạn (rất chậm).
4. Không còn 1.000 (điểm mâu thuẫn không thể đoán đúng cùng lúc với các điểm khác). Mô hình cho điểm đó
   một $p$ "lưng chừng" (gần 0.5) — nó *thỏa hiệp* để giảm tổng mất mát.
5. $z = 0.5$; $p \approx 0.622$; $r = 0.622 - 1 = -0.378$; $\partial J/\partial w = r\cdot x = -0.378$.

</details>

---

## 6. Tiếp theo (What's next)

Ta đã có hai mô hình (hồi quy và phân loại) dùng chung một khung, nhưng cả hai vẫn chỉ là **một đường
thẳng** — chúng bó tay với những quy luật cong, phức tạp. Ở **Chương 6 — Từ nơ-ron đến mạng (From a
neuron to a network)**, ta xếp chồng nhiều "đơn vị tuyến tính + hàm kích hoạt" thành một **mạng nơ-ron
nhiều tầng (multilayer perceptron — MLP)**, và thấy vì sao **tính phi tuyến (nonlinearity)** mở ra khả
năng học những quy luật mà một đường thẳng không bao giờ với tới.
