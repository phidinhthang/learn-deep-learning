# Chương 9 — Tensor và autograd (Tensors & autograd)

## 0. Chúng ta đang ở đâu (Where we are)

Hết Chương 8, bạn đã **tự tay** dựng trọn một pipeline deep learning bằng NumPy: forward, backward (lan
truyền ngược viết tay), cập nhật, chia dữ liệu, lô nhỏ, đánh giá — và đạt ~96–97% trên MNIST. Bạn hiểu
**từng bánh răng**.

Từ chương này ta bước sang **PyTorch** và làm **lại đúng những việc đó** — không phải vì NumPy sai, mà
vì ba lý do thực tế: (1) tự viết `backward` cho mỗi kiến trúc mới rất cực và dễ sai; (2) ta muốn chạy
**GPU**; (3) ta muốn tiến tới **CNN** (Chương 12+) mà backward của tích chập viết tay thì rất rối.

Trên bản đồ pipeline, **không ô nào thay đổi**. Khung "mô hình → mất mát → gradient → cập nhật" vẫn y
nguyên. Cái duy nhất được thay là: **autograd** tự tính phần gradient mà Chương 7 ta làm bằng tay.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Tensor: "mảng NumPy có thêm siêu năng lực"

**Tensor** của PyTorch gần như **giống hệt** `ndarray` của NumPy: cũng có shape, dtype, cũng cộng/trừ
theo phần tử, cũng `@` để nhân ma trận, cũng phát sóng (broadcasting). Hai siêu năng lực thêm vào:

1. **Chạy trên GPU.** Đặt tensor lên `"cuda"` thì mọi phép toán chạy trên GPU — nhanh gấp bội cho mạng
   lớn. Cùng một đoạn code chạy được cả CPU lẫn GPU (**device-agnostic**) nhờ một biến `device`.
2. **Tự tính gradient (autograd).** Nếu một tensor được đánh dấu `requires_grad=True`, PyTorch **ghi
   lại** mọi phép toán bạn làm với nó, để sau này tự suy ra đạo hàm.

### 1.2. Autograd làm đúng việc của Chương 7

Khi bạn tính toán với các tensor cần gradient, PyTorch âm thầm dựng một **đồ thị tính toán
(computational graph)**: ai được tính ra từ ai, bằng phép gì. Gọi `loss.backward()`, PyTorch đi **ngược**
đồ thị đó, nhân các đạo hàm cục bộ theo **quy tắc chuỗi** — **chính xác** thuật toán lan truyền ngược
bạn đã cài tay ở Chương 7, chỉ là **tự động** và cho **mọi** phép toán. Kết quả gradient được gắn vào
thuộc tính `.grad` của từng tham số.

> Nói cách khác: Chương 7 bạn **tự lái**; autograd là **số tự động**. Hộp số vẫn vậy, chỉ là máy sang số
> giúp bạn. Vì bạn đã biết bên trong, bạn sẽ không bao giờ coi autograd là phép thuật khó hiểu.

### 1.3. Ba thói quen mới phải nhớ

- **`requires_grad=True`** cho những tensor là **tham số cần học** (như `w`, `b`).
- **`loss.backward()`** để tính gradient; đọc ở `w.grad`, `b.grad`.
- **Xóa gradient sau mỗi bước** (`w.grad.zero_()`): autograd **cộng dồn** gradient qua các lần
  `backward`, nên nếu không xóa, bước sau sẽ cộng nhầm gradient của bước trước.

---

## 2. Toán học (The math)

### 2.1. Đồ thị tính toán + quy tắc chuỗi = autograd

Mọi biểu thức là một chuỗi phép cơ bản. Ví dụ $L = (wx + b - y)^2$ tách thành:

$$ z = wx + b \;\to\; r = z - y \;\to\; L = r^2. $$

Autograd biết đạo hàm cục bộ của từng mắt: $\frac{\partial L}{\partial r} = 2r$, $\frac{\partial r}{\partial z} = 1$,
$\frac{\partial z}{\partial w} = x$. Đi ngược và **nhân** lại (quy tắc chuỗi):

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial r}\cdot\frac{\partial r}{\partial z}\cdot
   \frac{\partial z}{\partial w} = 2r\cdot 1\cdot x = 2(wx + b - y)\,x. $$

Đúng **y hệt** công thức ta tự suy ở Chương 4! `loss.backward()` chỉ là làm phép nhân-ngược này một cách
tự động, cho đồ thị to cỡ nào cũng được.

### 2.2. Ví dụ kiểm chứng (tiny example)

Lấy $f(x) = x^3$, đạo hàm tay $f'(x) = 3x^2$, tại $x = 2$ cho $f'(2) = 12$.

```python
x = torch.tensor(2.0, requires_grad=True)
f = x ** 3
f.backward()          # autograd tính df/dx
# x.grad == 12.0  == 3 * 2^2
```

Autograd ra đúng 12 — khớp đạo hàm tay. Đây là bài kiểm tra "autograd = chain rule" cô đọng nhất.

### 2.3. Vì sao phải xóa gradient (gradient accumulation)

Mỗi lần gọi `.backward()`, PyTorch **cộng** kết quả vào `.grad` đang có:
$\text{grad} \mathrel{+}= \nabla_\theta L$. Thiết kế vậy để linh hoạt (vd cộng gradient từ nhiều phần),
nhưng trong vòng lặp huấn luyện thông thường ta muốn gradient **của riêng bước này**, nên phải `zero_()`
trước/sau mỗi bước. Quên xóa = gradient phình to dần = bước nhảy sai cỡ.

### 2.4. `torch.no_grad()` lúc cập nhật

Khi đã có gradient và muốn làm $w \leftarrow w - \eta\,\nabla_w L$, bản thân **phép cập nhật** không nên
bị autograd theo dõi (nó đâu phải một phần của mô hình). Bọc trong `with torch.no_grad():` để PyTorch
tạm ngừng ghi đồ thị — vừa đúng vừa tiết kiệm bộ nhớ.

---

## 3. Cài đặt (Implementation)

### 3.1. Tensor + autograd cơ bản — `ch09_autograd.py` (phần A, B)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"   # Chương 10 -> get_device()

a = torch.tensor([[1., 2.], [3., 4.]], device=device)
a + 10        # cộng từng phần tử (như NumPy)
a @ a         # nhân ma trận
a.cpu().numpy()   # đổi sang NumPy (đưa về CPU trước)

x = torch.tensor(2.0, requires_grad=True)
f = x ** 3
f.backward()
x.grad        # -> 12.0
```

### 3.2. Dựng lại hồi quy tuyến tính Chương 4 — `ch09_autograd.py` (phần C)

So với Chương 4 (NumPy): ta **bỏ** dòng tự tính `grad_w`, `grad_b`; thay bằng `loss.backward()`.

```python
torch.manual_seed(42)
xs = torch.tensor([-2., -1., 0., 1., 2.], device=device)
ys = torch.tensor([12., 16., 20., 24., 28.], device=device)
w = torch.zeros(1, requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)
lr = 0.1

for step in range(51):
    y_pred = w * xs + b                  # forward
    loss = ((y_pred - ys) ** 2).mean()   # MSE
    loss.backward()                      # AUTOGRAD thay cho backward viết tay
    with torch.no_grad():
        w -= lr * w.grad;  b -= lr * b.grad
        w.grad.zero_();    b.grad.zero_()
```

### 3.3. Dựng lại hồi quy logistic Chương 5 — `ch09_logreg_torch.py`

Y khung trên, chỉ đổi forward sang `sigmoid` và loss sang BCE — **không cần** tự suy gradient $(p-y)x$:

```python
for step in range(9):
    p = torch.sigmoid(w * x + b)
    loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()   # BCE
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad;  b -= lr * b.grad
        w.grad.zero_();    b.grad.zero_()
```

---

## 4. Chạy thử (Run it)

**(a) Tensor, autograd & hồi quy tuyến tính:**

```powershell
python ch09_autograd.py
```

**Kết quả mong đợi (expected output)** — *kết quả của bạn có thể hơi khác (your numbers may vary a
little)*; nếu máy có GPU thì dòng đầu in `device = cuda`:

```
device = cpu
a.shape = torch.Size([2, 2]) | a.dtype = torch.float32
a + 10  =
 tensor([[11., 12.],
        [13., 14.]])
a @ a   =
 tensor([[ 7., 10.],
        [15., 22.]])
sang numpy: [1. 2. 3. 4.]
df/dx tại x=2 = 12.0 (đúng 3*x^2 = 12)
step   0 | loss = 432.000000 | w = 0.0000, b = 0.0000
step  10 | loss = 4.612856 | w = 3.9758, b = 17.8525
step  20 | loss = 0.053169 | w = 3.9999, b = 19.7694
step  30 | loss = 0.000613 | w = 4.0000, b = 19.9752
step  40 | loss = 0.000007 | w = 4.0000, b = 19.9973
step  50 | loss = 0.000000 | w = 4.0000, b = 19.9997
Học xong: w = 4.0000, b = 19.9998  (quan hệ thật: w=4, b=20)
Dự đoán cho x=1.5: 25.9998  (đúng phải là 26)
```

So với output Chương 4: **giống hệt từng chữ số**. Đó là điều cốt lõi — autograd cho **đúng** gradient
mà ta tự suy bằng tay, nên quỹ đạo học không đổi.

**(b) Hồi quy logistic:**

```powershell
python ch09_logreg_torch.py
```

```
step 0 | loss = 0.6931 | w = 0.0000
step 2 | loss = 0.3548 | w = 0.8806
step 4 | loss = 0.2519 | w = 1.1758
step 6 | loss = 0.2098 | w = 1.3199
step 8 | loss = 0.1872 | w = 1.4090
Dự đoán   = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
Nhãn đúng  = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
```

Lại khớp tinh thần Chương 5: mất mát từ $\ln 2 \approx 0.6931$ giảm dần, $w$ tiến về ~1.4, $b$ ở lại
~0 (dữ liệu đối xứng), và phân loại đúng cả 6 điểm.

Đọc kết quả:

- **Cùng dữ liệu, cùng kết quả như Chương 4–5** — nhưng ta đã **xóa** đoạn tự viết gradient. Với mô hình
  to (CNN), tiết kiệm này là khổng lồ.
- **`device` tự chọn cpu/cuda:** cùng một file chạy được trên laptop lẫn máy có GPU mà không sửa gì.

---

## 5. Bài tập (Exercises)

1. **Quên xóa gradient.** Bỏ hai dòng `w.grad.zero_(); b.grad.zero_()` trong `ch09_autograd.py`. Quỹ đạo
   loss thay đổi thế nào? Giải thích bằng mục 2.3.
2. **Kiểm chứng autograd bằng tay.** Với `x = torch.tensor(3.0, requires_grad=True)` và `f = x**2 + 2*x`,
   tính tay `f'(3)` rồi so với `x.grad` sau `f.backward()`.
3. **Bỏ `no_grad`.** Thử cập nhật `w -= lr * w.grad` **không** bọc `torch.no_grad()`. PyTorch báo lỗi
   gì? (Gợi ý: nó không cho sửa tại chỗ một tensor đang được autograd theo dõi.)
4. **Tăng số bước logistic.** Cho `ch09_logreg_torch.py` chạy 50 bước thay vì 9. `w` có hội tụ về một
   số cố định không, hay cứ lớn dần? Vì sao (gợi ý: dữ liệu **tách rời hoàn toàn** — separable)?
5. **Đưa lên GPU (nếu có).** Nếu máy bạn có CUDA, xác nhận dòng đầu in `device = cuda` và kết quả vẫn
   như cũ. Nếu không có, giải thích vì sao code vẫn chạy bình thường trên CPU.

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. Gradient **cộng dồn** qua các bước → mỗi bước nhảy ngày một to → loss dao động hoặc phát tán thay vì
   giảm mượt. Đó đúng là lý do phải `zero_()`.
2. $f'(x) = 2x + 2 \Rightarrow f'(3) = 8$. `x.grad` phải in ra `8.0`.
3. Lỗi đại loại "a leaf Variable that requires grad is being used in an in-place operation". `no_grad`
   tắt việc theo dõi nên phép cập nhật tại chỗ trở nên hợp lệ.
4. `w` **cứ lớn dần** (không hội tụ về số cố định): với dữ liệu tách rời hoàn toàn, loss giảm mãi khi
   $w\to\infty$ (sigmoid càng dốc càng "chắc"). Đây là lý do thực tế người ta thêm **điều chuẩn
   (regularization)** để ghìm $w$.
5. Có CUDA: in `device = cuda`, kết quả số gần như y hệt (sai khác cực nhỏ do thứ tự tính trên GPU).
   Không CUDA: `torch.cuda.is_available()` trả `False` → `device = "cpu"` → mọi tensor nằm trên CPU,
   chạy bình thường.

</details>

---

## 6. Tiếp theo (What's next)

Ta đã có **tensor** và **autograd** — nhưng vẫn tự quản từng tham số `w`, `b` và tự viết bước cập nhật.
Ở **Chương 10 — Mô hình kiểu PyTorch (Models the PyTorch way)**, ta dùng `nn.Module` để gói cả mô hình
(không phải khai báo lẻ từng tham số), `torch.optim` để lo bước cập nhật (`optimizer.step()`), và
`Dataset`/`DataLoader` để thay `iterate_minibatches`. Ta cũng tạo `ddl/config.py` với `get_device()` và
đối tượng `Config` để gom mọi siêu tham số về một chỗ.
