# Chương 8 — Huấn luyện MLP & quy trình thực tế (Training the MLP + the real pipeline)

## 0. Chúng ta đang ở đâu (Where we are)

- **Chương 6–7:** ta có đủ bộ đồ nghề **forward + backward + cập nhật** cho một mạng nhiều tầng, và đã
  cho nó tự giải XOR — nhưng XOR chỉ có **4 điểm đồ chơi**.

Chương này là lần đầu ta chạm **dữ liệu thật: MNIST** (70.000 ảnh chữ số viết tay) và lấp nốt **những ô
pipeline còn bỏ trống** từ Chương 1:

> (1) thu thập → **(2) chia dữ liệu** → **(3) tiền xử lý** → (4) mô hình → (5) mất mát →
> **(6) huấn luyện theo lô** → **(7) đánh giá** → (suy luận).

Mô hình và toán **không đổi** so với Chương 7. Cái mới hoàn toàn nằm ở **quy trình quanh** việc huấn
luyện: chia train/val/test, chia **lô nhỏ (mini-batch)**, lặp nhiều **epoch**, và đo **độ chính xác**
trên dữ liệu chưa từng thấy. Đây chính là khung mà mọi dự án deep learning thật đều dùng.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. MNIST là gì, và vì sao phải "duỗi" ảnh

Mỗi ảnh MNIST là lưới $28\times28$ điểm ảnh (pixel) mức xám, mỗi pixel một số $0$ (đen) … $255$ (trắng).
Nhãn là chữ số đúng $0..9$. MLP của ta nhận **một vector** chứ không nhận lưới 2 chiều, nên ta **duỗi
(flatten)** mỗi ảnh thành vector dài $28\times28 = 784$. (Việc duỗi này **vứt mất** thông tin "pixel nào
cạnh pixel nào" — đó đúng là điểm yếu mà **CNN** ở Chương 12 sẽ sửa. Giờ cứ chấp nhận.)

### 1.2. Vì sao phải chia dữ liệu (train / val / test)?

Mục tiêu thật của học máy **không** phải nhớ thuộc dữ liệu đã thấy, mà là **tổng quát hóa
(generalization)** — đoán đúng trên dữ liệu **mới**. Nếu chấm điểm mô hình ngay trên dữ liệu nó vừa
học, ta bị lừa: nó có thể **học vẹt (overfitting)** mà ta không hay. Nên ta chia ba phần, mỗi phần một
vai:

| Tập | Vai trò | Mô hình có được "học" từ nó không? |
|---|---|---|
| **train** | để cập nhật trọng số | Có |
| **val** (validation) | theo dõi trong lúc huấn luyện, chỉnh siêu tham số (lr, số epoch…) | Không (chỉ để xem) |
| **test** | chấm điểm cuối cùng, **một lần** | Không, tuyệt đối |

Quy tắc vàng: **không bao giờ** để quyết định nào bị ảnh hưởng bởi tập test, nếu không con số cuối cùng
sẽ lạc quan giả tạo.

### 1.3. Vì sao chia lô nhỏ (mini-batch)?

Ở Chương 4–7 mỗi bước ta dùng **toàn bộ** dữ liệu để tính gradient. Với 54.000 ảnh, làm vậy mỗi bước
thì **chậm và ngốn bộ nhớ**. Giải pháp: mỗi bước chỉ lấy một **lô nhỏ** (vd 64 ảnh), tính gradient
**xấp xỉ** trên lô đó rồi cập nhật ngay. Ta được:

- **Nhanh hơn nhiều:** cập nhật rất nhiều lần trong một lượt dữ liệu, thay vì một lần.
- **Vừa bộ nhớ:** chỉ giữ 64 ảnh trong RAM mỗi lúc.
- **Nhiễu có ích:** gradient lô nhỏ hơi "rung", giúp thoát những chỗ kẹt nông.

Một **epoch** = duyệt **hết** tập train đúng **một lần** (qua tất cả các lô). Ta lặp vài epoch.

### 1.4. Tiền xử lý: chuẩn hóa (normalization / standardization)

Pixel gốc nằm trong $0..255$; ta chia 255 để về $[0,1]$ — đó là **chuẩn hóa (normalization)** đơn giản,
giúp các con số nhỏ và đồng đều, gradient descent ổn định hơn. Một họ hàng tổng quát hơn là **chuẩn hóa
chuẩn tắc (standardization)**: trừ trung bình, chia độ lệch chuẩn để mỗi đặc trưng có mean 0, std 1
(lớp `Standardizer`, mục 3). Với MNIST chia 255 là đủ; `Standardizer` hữu ích hơn cho dữ liệu bảng có
các cột thang đo lệch nhau.

---

## 2. Toán học (The math)

### 2.1. Gradient lô nhỏ là ước lượng không chệch của gradient toàn phần

Mất mát toàn phần là trung bình trên $N$ mẫu: $J = \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_i$. Gradient của
nó là trung bình các gradient từng mẫu. Nếu lấy một lô **ngẫu nhiên** $B$ gồm $|B|$ mẫu và tính

$$ \nabla_\theta J_B = \frac{1}{|B|}\sum_{i\in B}\nabla_\theta \mathcal{L}_i, $$

thì **kỳ vọng** của $\nabla_\theta J_B$ đúng bằng gradient toàn phần $\nabla_\theta J$. Nói cách khác,
gradient lô nhỏ là một **ước lượng không chệch (unbiased estimate)**: trung bình thì đúng hướng, chỉ
hơi "rung" quanh hướng thật. Đó là nền tảng lý thuyết khiến **SGD theo lô nhỏ** hoạt động.

### 2.2. Số bước cập nhật

Với $N_\text{train}$ mẫu, cỡ lô $|B|$, mỗi epoch có $\lceil N_\text{train}/|B|\rceil$ bước cập nhật. Ví
dụ $N_\text{train}=54{,}000$, $|B|=64$: mỗi epoch $\approx 844$ bước; 5 epoch $\approx 4220$ lần cập
nhật trọng số — nhiều hơn hẳn so với 5 lần nếu dùng toàn bộ dữ liệu. Đó là lý do lô nhỏ học nhanh hơn
theo "thời gian thực".

### 2.3. Độ chính xác (accuracy)

Mất mát cross-entropy là thứ ta **tối ưu**, nhưng con người muốn đọc **độ chính xác**: dự đoán nhãn
bằng $\hat{c}_i = \arg\max_k \text{logits}_{i,k}$ (lớp có điểm cao nhất), rồi

$$ \text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{c}_i = y_i]. $$

Để ý: ta **không cần** softmax để chọn nhãn — $\arg\max$ của logits và của softmax(logits) là một (xem
bài tập 4).

---

## 3. Cài đặt (Implementation)

### 3.1. Nạp & chia dữ liệu — `ddl/data.py`

```python
def load_mnist(flatten=True, normalize=True):
    from torchvision import datasets   # CHỈ mượn để TẢI ảnh thô về
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)
    X_train = train.data.numpy().astype(np.float32)    # (60000, 28, 28)
    y_train = train.targets.numpy().astype(np.int64)   # (60000,)
    X_test = test.data.numpy().astype(np.float32)
    y_test = test.targets.numpy().astype(np.int64)
    if normalize:
        X_train, X_test = X_train / 255.0, X_test / 255.0   # pixel -> [0, 1]
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)     # -> (60000, 784)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return (X_train, y_train), (X_test, y_test)
```

`train_val_test_split` xáo trộn theo seed rồi cắt; `iterate_minibatches` sinh từng lô:

```python
def iterate_minibatches(X, y, batch_size, shuffle=True, seed=None):
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)   # xáo lại mỗi epoch
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]
```

### 3.2. Đo lường — `ddl/metrics.py`

```python
def predict_labels(logits):
    return np.argmax(logits, axis=-1)          # lớp có logit cao nhất

def accuracy(y_pred_labels, y_true):
    return np.mean(y_pred_labels == y_true)    # tỉ lệ đoán đúng
```

### 3.3. Chuẩn hóa tổng quát — `ddl/preprocessing.py`

```python
class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8       # +eps để không chia cho 0
        return self
    def transform(self, X):       return (X - self.mean_) / self.std_
    def fit_transform(self, X):   return self.fit(X).transform(X)
    def inverse_transform(self, X): return X * self.std_ + self.mean_
```

> Quan trọng: gọi `fit` **chỉ trên train**, rồi `transform` cho cả val/test bằng cùng `mean_`, `std_`
> đó. Học thống kê từ val/test là **rò rỉ dữ liệu (data leakage)**.

### 3.4. Ráp toàn bộ pipeline — `ch08_train_mnist.py`

```python
(X_train_full, y_train_full), (X_test, y_test) = load_mnist(flatten=True, normalize=True)
(X_train, y_train), (X_val, y_val), _ = train_val_test_split(
    X_train_full, y_train_full, val_fraction=0.1, seed=42)

net = Sequential([Linear(784, 128, seed=42), ReLU(), Linear(128, 10, seed=43)])
loss_fn = SoftmaxCrossEntropy()
lr, n_epochs, batch_size = 0.1, 5, 64

for epoch in range(1, n_epochs + 1):
    for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
        logits = net.forward(X_batch)
        loss = loss_fn.forward(logits, y_batch)
        net.backward(loss_fn.backward())          # forward -> loss -> backward ...
        sgd_step(net.parameters(), lr)            # ... -> update, đúng khung Chương 7
    val_acc = accuracy(predict_labels(net.forward(X_val)), y_val)
```

Để ý: **năm dòng lõi** (forward → loss → backward → step) **giống hệt** Chương 7. Tất cả phần thêm chỉ
là *vòng lặp lô* và *đánh giá* bao quanh.

---

## 4. Chạy thử (Run it)

```powershell
python ch08_train_mnist.py
```

Lần đầu chạy sẽ tự tải MNIST (~12 MB) về `./data`. **Kết quả mong đợi (expected output)** — *kết quả
của bạn có thể hơi khác (your numbers may vary a little)*; độ chính xác test thường rơi vào khoảng
**96–97%**:

```
train=(54000, 784), val=(6000, 784), test=(10000, 784)
epoch 1 | train_loss = 0.4129 | val_acc = 0.9357
epoch 2 | train_loss = 0.2086 | val_acc = 0.9482
epoch 3 | train_loss = 0.1607 | val_acc = 0.9583
epoch 4 | train_loss = 0.1313 | val_acc = 0.9621
epoch 5 | train_loss = 0.1108 | val_acc = 0.9655
Độ chính xác trên TEST = 0.9648
```

Đọc kết quả:

- **train_loss giảm đều, val_acc tăng đều:** mô hình đang học **và** tổng quát hóa tốt (val là dữ liệu
  nó không cập nhật từ đó). Nếu val_acc bắt đầu **tụt** trong khi train vẫn lên, đó là dấu hiệu **học
  vẹt (overfitting)** — ta sẽ chống lại bằng **dropout** ở Chương 14.
- **~96–97% trên test:** một mạng nơ-ron **viết tay hoàn toàn bằng NumPy**, không thư viện deep learning
  nào, đã đọc được chữ số viết tay đúng 24/25 lần. Toàn bộ máy móc bên dưới chính là thứ bạn tự dựng từ
  Chương 6–7.

> **Cột mốc:** đây là đỉnh của "kỷ nguyên NumPy". Bạn đã đi trọn pipeline trên dữ liệu thật, tự tay từng
> bánh răng. Từ Chương 9, ta làm **lại đúng những việc này** bằng PyTorch — không phải vì NumPy sai, mà
> để **tự động hóa** phần `backward` và mở đường tới CNN/GPU.

---

## 5. Bài tập (Exercises)

1. **Đổi cỡ lô.** Chạy với `batch_size = 8`, rồi `256`. Cái nào mỗi epoch chậm hơn (nhiều bước hơn)?
   Cái nào val_acc mượt hơn? Liên hệ "nhiễu lô nhỏ" ở mục 1.3.
2. **Tăng/giảm số epoch.** Đặt `n_epochs = 15`. val_acc có tiếp tục tăng mãi không, hay chững lại? Có
   thấy dấu hiệu học vẹt (train_loss vẫn giảm nhưng val_acc đứng/tụt) không?
3. **Rộng hơn.** Đổi tầng ẩn từ 128 thành 256 nơ-ron (`Linear(784, 256)` và `Linear(256, 10)`). Độ
   chính xác test thay đổi ra sao? Đánh đổi gì (thời gian/bộ nhớ)?
4. **Softmax có cần cho dự đoán không?** Chứng minh bằng lời: vì sao `argmax(logits)` luôn cho cùng nhãn
   với `argmax(softmax(logits))`? (Gợi ý: softmax tăng đơn điệu.)
5. **Rò rỉ dữ liệu.** Giả sử ai đó `fit` Standardizer trên **toàn bộ** train+test rồi mới chia. Vì sao
   điều này khiến con số test lạc quan giả tạo?

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. `batch_size=8`: mỗi epoch **nhiều bước** hơn (~6750 bước) → chậm hơn mỗi epoch nhưng cập nhật dày;
   val_acc "rung" hơn. `256`: ít bước, mỗi epoch nhanh, đường val_acc mượt nhưng mỗi epoch tiến chậm.
2. val_acc tăng rồi **chững lại** (bão hòa); kéo dài nữa dễ thấy train_loss vẫn giảm mà val_acc đứng —
   đó là học vẹt. "Điểm dừng tốt" thường là nơi val_acc thôi cải thiện (early stopping).
3. Thường **nhích lên** một chút (~97%), đổi lại tốn thêm thời gian và bộ nhớ. Lợi ích giảm dần.
4. softmax là hàm **tăng đơn điệu** theo từng logit và giữ nguyên thứ tự, nên phần tử lớn nhất trước và
   sau softmax là **cùng một vị trí** → `argmax` không đổi. Vì vậy dự đoán nhãn không cần softmax.
5. Vì `mean_`, `std_` khi đó "đã thấy" cả test → mô hình gián tiếp biết thông tin của test trước khi
   chấm điểm → điểm test cao hơn thực tế nó sẽ đạt trên dữ liệu hoàn toàn mới.

</details>

---

## 6. Tiếp theo (What's next)

Bạn vừa hoàn tất **trọn vẹn một pipeline deep learning bằng tay** trên dữ liệu thật — đỉnh của phần
NumPy. Ở **Chương 9 — Tensor và autograd (Tensors & autograd)**, ta bước sang **PyTorch**: gặp **tensor**
(họ hàng của mảng NumPy nhưng biết chạy GPU và **tự tính gradient**), và để **autograd** làm thay đúng
phần `backward` ta vừa cặm cụi viết tay — rồi dựng lại hồi quy Chương 4–5 chỉ trong vài dòng.
