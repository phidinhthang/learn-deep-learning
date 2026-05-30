# Chương 7 — Lan truyền ngược bằng tay (Backpropagation by hand)

## 0. Chúng ta đang ở đâu (Where we are)

- **Chương 4–5:** một đường thẳng tự học bằng gradient descent (forward → loss → gradient → update).
- **Chương 6:** xếp chồng nhiều tầng + ReLU thành **mạng (MLP)** — nhưng ta **đặt tay** trọng số vì
  đã biết đáp án XOR. Câu hỏi treo lại: làm sao mạng **tự học** trọng số đó?

Đây là chương quan trọng nhất khóa học. Trên bản đồ pipeline ta vẫn ở ô (6) **vòng lặp huấn luyện**,
nhưng lần đầu phải tính gradient cho một hàm **nhiều tầng lồng nhau**. Công cụ để làm điều đó là **lan
truyền ngược (backpropagation)** — thực chất chỉ là **quy tắc chuỗi (chain rule, Chương 3) áp một cách
có tổ chức**, đi ngược từ mất mát về từng tham số. Cuối chương, mạng Chương 6 sẽ **tự giải XOR từ con
số 0**.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Vấn đề: gradient của một hàm lồng nhiều lớp

Gradient descent cần một thứ: **đạo hàm của mất mát theo từng tham số** ($\partial J/\partial W^{[1]}$,
$\partial J/\partial b^{[1]}$, …). Ở Chương 4 việc này dễ vì mô hình chỉ một tầng. Giờ mất mát là một
chuỗi **lồng nhau**:

$$ X \xrightarrow{W^{[1]}} Z^{[1]} \xrightarrow{\text{ReLU}} A^{[1]} \xrightarrow{W^{[2]}}
   \text{logits} \xrightarrow{\text{softmax+CE}} J. $$

$W^{[1]}$ ảnh hưởng tới $J$ **qua rất nhiều bước trung gian**. Tính trực tiếp $\partial J/\partial W^{[1]}$
bằng một công thức khổng lồ thì khủng khiếp. May thay, ta không cần.

### 1.2. Ý tưởng cốt lõi: chuyền "trách nhiệm" ngược về

**Quy tắc chuỗi** nói: đạo hàm qua một chuỗi = **nhân** các đạo hàm của từng mắt xích. Backprop khai
thác điều đó theo một mẹo tổ chức cực gọn:

> Mỗi tầng nhận từ tầng **sau** nó một con số "đầu ra của tôi sai bao nhiêu" — gọi là **gradient
> thượng nguồn (upstream gradient)** $\partial J/\partial(\text{đầu ra của tầng})$. Từ đó, mỗi tầng tự
> làm **hai việc**: (a) tính gradient cho **tham số của chính nó** (để cập nhật), và (b) tính gradient
> cho **đầu vào của nó**, rồi **chuyền tiếp ngược** cho tầng trước.

Cứ thế "trách nhiệm" lan từ mất mát (cuối) về tới tầng đầu. Mỗi tầng chỉ cần biết **đạo hàm cục bộ
(local derivative)** của riêng nó — không cần biết gì về phần còn lại của mạng. Đó là lý do ta cài
`backward` **riêng cho từng lớp** ở Chương 6–7.

### 1.3. Một phép loại suy

Hình dung một dây chuyền: nguyên liệu $X$ qua nhiều máy, ra sản phẩm, rồi bị chấm điểm lỗi $J$. Muốn
biết **chỉnh núm trên máy số 1** ảnh hưởng điểm lỗi thế nào, ta truyền thông tin "lỗi" ngược dây chuyền:
máy cuối báo cho máy trước "đầu ra của anh lệch chừng này", máy đó quy ra "vậy núm của tôi nên xoay
chừng này, và đầu vào của tôi lệch chừng kia" rồi báo tiếp ngược lên. Backprop chính là vậy.

---

## 2. Toán học (The math)

Mạng cho phần này: $D$ đầu vào → $H$ nơ-ron ẩn (ReLU) → $K$ logits, rồi softmax + cross-entropy.

$$ Z^{[1]} = X W^{[1]} + \mathbf{b}^{[1]}, \quad A^{[1]} = \text{ReLU}(Z^{[1]}), \quad
   \text{logits} = A^{[1]} W^{[2]} + \mathbf{b}^{[2]}, \quad J = \text{CE}(\text{softmax}(\text{logits}), y). $$

Ký hiệu tắt: với mọi đại lượng $U$, viết $\mathrm{d}U \equiv \partial J/\partial U$ (gradient của mất
mát theo $U$, **cùng shape với $U$**). Ta đi **ngược** từ $J$.

### 2.1. Mắt xích cuối: softmax + cross-entropy

Với một mẫu, $p_k = \text{softmax}(\text{logits})_k$ và mất mát $\mathcal{L} = -\log p_c$ ($c$ = lớp
đúng). Đạo hàm theo logit thứ $k$ (kết quả kinh điển, đẹp đến bất ngờ):

$$ \frac{\partial \mathcal{L}}{\partial\, \text{logit}_k} = p_k - \mathbb{1}[k = c]. $$

Tức "xác suất dự đoán **trừ** đáp án đúng dạng one-hot". Gom cả batch $N$ mẫu và vì $J$ là **trung
bình**:

$$ \boxed{\;\mathrm{d}\,\text{logits} = \frac{1}{N}\,(P - Y_{\text{onehot}})\;}\qquad \text{shape } (N, K). $$

<details>
<summary>Vì sao ra <code>p_k − 1[k=c]</code> (chứng minh ngắn)</summary>

$\mathcal{L} = -\log p_c$ nên $\dfrac{\partial \mathcal{L}}{\partial \text{logit}_k}
= -\dfrac{1}{p_c}\dfrac{\partial p_c}{\partial \text{logit}_k}$. Với softmax:
$\dfrac{\partial p_c}{\partial \text{logit}_k} = p_c(\mathbb{1}[k=c] - p_k)$. Thay vào:
$-\dfrac{1}{p_c}\,p_c(\mathbb{1}[k=c]-p_k) = p_k - \mathbb{1}[k=c]$. ∎
</details>

### 2.2. Qua tầng tuyến tính thứ hai

logits $= A^{[1]} W^{[2]} + \mathbf{b}^{[2]}$. Biết $\mathrm{d}\,\text{logits}$, áp quy tắc chuỗi cho
phép nhân ma trận (suy ra từ việc khớp shape — xem Chương 2):

$$ \mathrm{d}W^{[2]} = (A^{[1]})^\top\, \mathrm{d}\,\text{logits}, \qquad
   \mathrm{d}\mathbf{b}^{[2]} = \sum_{\text{batch}} \mathrm{d}\,\text{logits}, \qquad
   \mathrm{d}A^{[1]} = \mathrm{d}\,\text{logits}\,(W^{[2]})^\top. $$

Hai cái đầu để **cập nhật** $W^{[2]}, \mathbf{b}^{[2]}$; cái cuối là gradient **chuyền ngược** cho tầng
trước.

### 2.3. Qua ReLU

$A^{[1]} = \text{ReLU}(Z^{[1]})$. Đạo hàm ReLU theo từng phần tử là $1$ ở chỗ dương, $0$ ở chỗ còn lại.
Nên gradient chỉ "đi qua" tại những ô từng dương:

$$ \mathrm{d}Z^{[1]} = \mathrm{d}A^{[1]} \odot \mathbb{1}[Z^{[1]} > 0] \qquad (\odot:\text{nhân từng phần tử}). $$

### 2.4. Qua tầng tuyến tính thứ nhất

$Z^{[1]} = X W^{[1]} + \mathbf{b}^{[1]}$ — **cùng dạng** mục 2.2, chỉ thay tên:

$$ \mathrm{d}W^{[1]} = X^\top\, \mathrm{d}Z^{[1]}, \qquad
   \mathrm{d}\mathbf{b}^{[1]} = \sum_{\text{batch}} \mathrm{d}Z^{[1]}, \qquad
   \mathrm{d}X = \mathrm{d}Z^{[1]}\,(W^{[1]})^\top. $$

$\mathrm{d}X$ thường không cần (ta đâu chỉnh dữ liệu), nhưng nếu còn tầng trước nữa thì nó là gradient
chuyền tiếp. **Để ý quy luật lặp lại:** một tầng `Linear` luôn có cùng ba công thức $\mathrm{d}W,
\mathrm{d}b, \mathrm{d}x$ — đó chính là cái ta gói gọn trong `Linear.backward`.

### 2.5. Ví dụ tính tay một bước backward của `Linear`

Cho $x = \begin{bmatrix}1 & 2\end{bmatrix}$ ($N=1$, in$=2$), $W = \begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}$,
$\mathbf{b} = [0, 0]$. Forward: $y = xW + b = [1, 2]$. Giả sử tầng sau chuyền về
$\mathrm{d}y = [0.5, -1.0]$. Khi đó:

$$ \mathrm{d}W = x^\top \mathrm{d}y = \begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}0.5 & -1.0\end{bmatrix}
   = \begin{bmatrix}0.5 & -1.0\\ 1.0 & -2.0\end{bmatrix},\quad
   \mathrm{d}\mathbf{b} = [0.5, -1.0],\quad
   \mathrm{d}x = \mathrm{d}y\,W^\top = [0.5, -1.0]. $$

Khớp đúng các dòng trong `Linear.backward` — bạn có thể kiểm lại bằng `print` khi chạy.

---

## 3. Cài đặt (Implementation)

### 3.1. Thêm `backward` cho mỗi lớp — `ddl/numpy_nn.py`

Ta bổ sung đúng các công thức trên vào file Chương 6. Mỗi tầng nhận `grad` (gradient thượng nguồn), tự
tính gradient tham số (`self.dW`, `self.db`) và **trả về** gradient cho đầu vào để chuyền ngược tiếp.

```python
class Linear:
    def backward(self, grad):
        # grad = dJ/dy, shape (N, out). Áp quy tắc chuỗi:
        self.dW = self.x.T @ grad           # dJ/dW = x^T @ grad
        self.db = grad.sum(axis=0)          # dJ/db = cộng dồn theo batch
        return grad @ self.W.T              # dJ/dx = grad @ W^T  (chuyền ngược)
    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU:
    def backward(self, grad):
        return grad * self.mask             # gradient chỉ chảy qua chỗ từng dương
    def parameters(self):
        return []

class Sequential:
    def backward(self, grad):
        for layer in reversed(self.layers): # NGƯỢC: tầng cuối -> tầng đầu
            grad = layer.backward(grad)
        return grad
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
```

### 3.2. Mất mát softmax + cross-entropy — `SoftmaxCrossEntropy`

```python
class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)   # trừ max cho ổn định số
        exp = np.exp(z)
        self.p = exp / exp.sum(axis=1, keepdims=True)    # xác suất (N, K)
        self.y = y
        N = logits.shape[0]
        return -np.log(self.p[np.arange(N), y] + 1e-12).mean()
    def backward(self):
        N = self.y.shape[0]
        grad = self.p.copy()
        grad[np.arange(N), self.y] -= 1.0                # (p - onehot(y))
        return grad / N                                  # chia N vì loss là trung bình
```

### 3.3. Bước cập nhật — `sgd_step`

```python
def sgd_step(params, lr):
    for param, grad in params:
        param -= lr * grad      # theta <- theta - lr * grad (cập nhật tại chỗ)
```

### 3.4. Huấn luyện XOR — `ch07_backprop.py`

Vòng lặp giống hệt Chương 4–5, chỉ thêm một dòng `net.backward(grad)`:

```python
net = Sequential([Linear(2, 8, seed=42), ReLU(), Linear(8, 2, seed=43)])
loss_fn = SoftmaxCrossEntropy()
lr, n_epochs = 1.0, 2000

for epoch in range(1, n_epochs + 1):
    logits = net.forward(X)               # (1) forward
    loss = loss_fn.forward(logits, y)     # (2) đo mất mát
    grad = loss_fn.backward()             # (3) dJ/dlogits
    net.backward(grad)                    # (4) lan truyền ngược -> điền dW, db mọi tầng
    sgd_step(net.parameters(), lr)        # (5) cập nhật tham số
```

---

## 4. Chạy thử (Run it)

```powershell
python ch07_backprop.py
```

**Kết quả mong đợi (expected output)** — mạng học từ ngẫu nhiên nên *kết quả của bạn có thể hơi khác
(your numbers may vary a little)*, nhưng xu hướng luôn là: mất mát giảm về gần 0 và độ chính xác đạt
1.0.

```
epoch    1 | loss = 0.7012
epoch  400 | loss = 0.0181
epoch  800 | loss = 0.0070
epoch 1200 | loss = 0.0042
epoch 1600 | loss = 0.0030
epoch 2000 | loss = 0.0023
Dự đoán   = [0 1 1 0]
XOR đúng  = [0 1 1 0]
Độ chính xác (accuracy) = 1.0
```

Đọc kết quả:

- **Mất mát ~0.70 → ~0.00:** mạng thực sự đang học. Mốc khởi đầu $\approx \ln 2 \approx 0.693$ là mức
  "đoán mò 50–50" cho 2 lớp; giảm sâu hơn nghĩa là mạng ngày càng **tự tin và đúng**.
- **Dự đoán `[0 1 1 0]` = XOR:** mạng **tự tìm ra** một bộ trọng số giải XOR — **không ai mách** cho nó
  như Chương 6. Đây chính là câu trả lời cho câu hỏi treo cuối Chương 6.
- Bộ trọng số nó tìm được **không nhất thiết** giống bộ ta đặt tay ở Chương 6 — có vô số lời giải, và
  gradient descent rơi vào một trong số đó.

> **Khoảnh khắc cốt lõi:** toàn bộ những gì ta vừa làm — forward, đo loss, backward, cập nhật — là
> **đúng một thuật toán** dùng cho **mọi** mạng nơ-ron, từ XOR 9 tham số tới mạng triệu tham số. Từ
> Chương 9 trở đi, PyTorch sẽ **tự động hóa** bước `backward` này (gọi là **autograd**) — nhưng giờ
> bạn đã biết bên trong nó làm gì.

---

## 5. Bài tập (Exercises)

1. **Kiểm chứng ví dụ tay.** Thêm vài dòng tạo `lin = Linear(2, 2, seed=0)`, gán tay
   `lin.W = np.eye(2)`, `lin.b = np.zeros(2)`, chạy `lin.forward(np.array([[1., 2.]]))` rồi
   `lin.backward(np.array([[0.5, -1.0]]))`. In `lin.dW`, `lin.db` và so với mục 2.5.
2. **Bỏ ReLU.** Đổi mạng thành `Sequential([Linear(2, 8), Linear(8, 2)])` (không phi tuyến). Huấn
   luyện lại. Độ chính xác có lên 1.0 không? Giải thích bằng chứng minh ở Chương 6 mục 2.2.
3. **Đổi tốc độ học.** Thử `lr = 0.1` và `lr = 5.0`. Mô tả: cái nào hội tụ chậm, cái nào dao động/phát
   tán? Liên hệ "cái bát" ở Chương 4.
4. **Số nơ-ron ẩn.** Giảm tầng ẩn xuống `Linear(2, 2)`. Mạng còn giải được XOR không? (Gợi ý: Chương 6
   chỉ cần 2 nơ-ron ẩn — nhưng học **tự động** đôi khi cần dư ra một chút để dễ thoát kẹt.)
5. **Vì sao `grad / N`?** Nếu bỏ phép chia `N` trong `SoftmaxCrossEntropy.backward`, gradient sẽ lớn
   gấp $N$ lần. Điều đó tương đương với việc đổi gì trong vòng lặp? (Gợi ý: tốc độ học hiệu dụng.)

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. Phải in ra đúng `dW = [[0.5, -1.0], [1.0, -2.0]]`, `db = [0.5, -1.0]`.
2. Không. Không phi tuyến → hai `Linear` gập thành một phép tuyến tính → lại là "đường thẳng", chỉ đạt
   ~50–75% (không tách được XOR). Đúng như chứng minh tuyến tính-chồng-tuyến-tính.
3. `lr = 0.1`: hội tụ nhưng chậm (cần nhiều epoch hơn). `lr = 5.0`: bước quá dài, loss dao động hoặc
   tăng vọt (phát tán). Vùng `lr` "vừa phải" mới ổn định.
4. Thường **vẫn** giải được, nhưng nhạy với khởi tạo hơn (đôi khi kẹt ở ~75%). Nhiều nơ-ron ẩn cho
   mạng nhiều "đường thoát" hơn.
5. Bỏ chia $N$ = nhân gradient với $N$ = giống như dùng tốc độ học lớn gấp $N$ lần. Với $N=4$ có thể
   vẫn chạy, nhưng với batch lớn (MNIST, $N=64$) sẽ làm bước nhảy quá cỡ → bất ổn.

</details>

---

## 6. Tiếp theo (What's next)

Ta đã có đủ bộ đồ nghề **forward + backward + cập nhật** cho một mạng thật. Ở **Chương 8 — Huấn luyện
MLP & quy trình thực tế (Training the MLP + the real pipeline)**, ta rời bài toán đồ chơi để gặp dữ
liệu **thật đầu tiên — MNIST** (chữ số viết tay), và lấp nốt những ô pipeline còn bỏ trống: **chia dữ
liệu (train/val/test split)**, **lô nhỏ (mini-batch)**, **epoch**, và đo **độ chính xác** trên dữ liệu
chưa từng thấy.
