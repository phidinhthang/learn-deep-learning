# Chương 6 — Từ nơ-ron đến mạng (From a neuron to a network)

## 0. Chúng ta đang ở đâu (Where we are)

Chương 4 và 5 cho ta hai mô hình mạnh nhưng **cùng một giới hạn**: chúng đều chỉ là **một đường thẳng**
($z = wx + b$). Ranh giới chúng vẽ ra luôn thẳng — nên có những quy luật chúng **không bao giờ** học
được. Chương này phá vỡ giới hạn đó bằng cách **xếp chồng nhiều đơn vị tuyến tính + một hàm phi tuyến**
thành một **mạng nơ-ron nhiều tầng (multilayer perceptron — MLP)**.

Trên bản đồ pipeline, ta vẫn đang ở ô (4) **xây mô hình** — cụ thể là **lượt tính xuôi (forward pass)**
của một mạng. Ta **chưa** huấn luyện nó (cách *học* trọng số tự động là **lan truyền ngược** ở Chương 7);
ở đây ta đặt tay trọng số để chứng minh một điều: **mạng có thể biểu diễn những quy luật mà đường thẳng
chịu thua.**

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Một nơ-ron là gì?

Một **nơ-ron (neuron)** chỉ là thứ ta đã làm ở Chương 5: lấy đầu vào, tính tổng có trọng số $z = \mathbf{w}
\cdot \mathbf{x} + b$, rồi cho qua một **hàm kích hoạt (activation function)** $f$:

$$ a = f(\mathbf{w}\cdot\mathbf{x} + b). $$

Hồi quy logistic chính là **một** nơ-ron với $f = \sigma$ (sigmoid). Ý tưởng của mạng: dùng **nhiều**
nơ-ron, và cho đầu ra của lớp này làm đầu vào của lớp sau.

### 1.2. Giới hạn của đường thẳng: bài toán XOR

XOR ("hoặc loại trừ") cho nhãn **1** khi *đúng một* trong hai đầu vào bằng 1:

```
 x2
  1 | (0,1)→1          (1,1)→0
    |
  0 | (0,0)→0          (1,0)→1
    +----------------------- x1
        0                1
```

Hãy thử kẻ **một** đường thẳng tách các điểm **1** (gồm $(0,1)$ và $(1,0)$) khỏi các điểm **0** (gồm
$(0,0)$ và $(1,1)$). Bạn sẽ thấy **không thể** — hai lớp nằm chéo nhau. XOR **không tách rời tuyến tính
được (not linearly separable)**. Một nơ-ron đơn (một đường thẳng) bó tay.

### 1.3. Vì sao cần tính phi tuyến (nonlinearity)?

"Vậy thì xếp nhiều tầng tuyến tính lên nhau là xong?" — **Không.** Đây là cái bẫy quan trọng nhất của
chương: **ghép hai phép tuyến tính vẫn ra một phép tuyến tính** (ta chứng minh ở mục 2.2). Mười tầng
tuyến tính chồng lên nhau vẫn chỉ là... một đường thẳng.

Thứ phá vỡ điều đó là một **hàm kích hoạt phi tuyến** đặt **xen giữa** các tầng — như **ReLU**:
$\text{ReLU}(z) = \max(0, z)$. Cái "gấp khúc" tại 0 của ReLU cho phép mạng **bẻ cong** không gian, ghép
nhiều đoạn thẳng lại thành ranh giới phức tạp tùy ý. Với XOR, chỉ cần **2 nơ-ron ẩn + ReLU** là đủ.

> Hình dung trực giác: mỗi nơ-ron ẩn học **một đặc trưng (feature)** đơn giản (một "nửa mặt phẳng"). Tầng
> sau **kết hợp** các đặc trưng đó. Nhiều nơ-ron + phi tuyến = ghép được những hình rất phức tạp từ các
> mảnh đơn giản.

---

## 2. Toán học (The math)

### 2.1. Tầng (layer) dưới dạng ma trận

Gom $H$ nơ-ron thành **một tầng**: thay vì một vector trọng số, ta có **ma trận** $W$. Cho cả lô $N$ mẫu:

$$ Z = X W + \mathbf{b}, \qquad X:(N, D),\; W:(D, H),\; \mathbf{b}:(H,),\; Z:(N, H). $$

(Đây đúng là phép nhân ma trận của Chương 2 — mỗi mẫu đi qua $W$ để ra $H$ con số.)

### 2.2. Vì sao tuyến tính chồng tuyến tính vẫn là tuyến tính (chứng minh)

Cho hai tầng tuyến tính $f_1(\mathbf{x}) = W_1\mathbf{x} + \mathbf{b}_1$ và $f_2(\mathbf{h}) = W_2\mathbf{h}
+ \mathbf{b}_2$. Ghép lại:

$$ f_2(f_1(\mathbf{x})) = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
   = \underbrace{(W_2 W_1)}_{W'}\,\mathbf{x} + \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\mathbf{b}'}
   = W'\mathbf{x} + \mathbf{b}'. $$

Vẫn là **một** phép tuyến tính $W'\mathbf{x} + \mathbf{b}'$. Đó là lý do **bắt buộc** phải có hàm phi
tuyến xen giữa, nếu không thêm tầng chẳng tăng thêm sức mạnh nào.

### 2.3. Lượt tính xuôi của MLP 2 tầng

Mạng ta dùng cho XOR: $D=2$ đầu vào → $H=2$ nơ-ron ẩn (ReLU) → 1 đầu ra.

$$ Z^{[1]} = X W^{[1]} + \mathbf{b}^{[1]}, \quad A^{[1]} = \text{ReLU}(Z^{[1]}), \quad
   \hat{Y} = A^{[1]} W^{[2]} + \mathbf{b}^{[2]}. $$

| Đại lượng | Shape | Ý nghĩa |
|---|---|---|
| $X$ | $(N, 2)$ | đầu vào |
| $W^{[1]}, \mathbf{b}^{[1]}$ | $(2, 2), (2,)$ | trọng số tầng ẩn |
| $A^{[1]}$ | $(N, 2)$ | đầu ra tầng ẩn (sau ReLU) |
| $W^{[2]}, \mathbf{b}^{[2]}$ | $(2, 1), (1,)$ | trọng số tầng ra |
| $\hat{Y}$ | $(N, 1)$ | dự đoán |

### 2.4. Một bộ trọng số giải XOR (tính tay)

Chọn:

$$ W^{[1]} = \begin{bmatrix}1&1\\1&1\end{bmatrix},\;\mathbf{b}^{[1]}=[0,\,-1], \qquad
   W^{[2]} = \begin{bmatrix}1\\-2\end{bmatrix},\;\mathbf{b}^{[2]}=[0]. $$

Tức $h_1 = \text{ReLU}(x_1 + x_2)$, $h_2 = \text{ReLU}(x_1 + x_2 - 1)$, và $\hat{y} = h_1 - 2h_2$. Thử
hết 4 điểm:

| $(x_1, x_2)$ | $h_1$ | $h_2$ | $\hat{y} = h_1 - 2h_2$ | XOR |
|---|---|---|---|---|
| $(0,0)$ | $\text{ReLU}(0)=0$ | $\text{ReLU}(-1)=0$ | $0$ | 0 ✓ |
| $(0,1)$ | $\text{ReLU}(1)=1$ | $\text{ReLU}(0)=0$ | $1$ | 1 ✓ |
| $(1,0)$ | $\text{ReLU}(1)=1$ | $\text{ReLU}(0)=0$ | $1$ | 1 ✓ |
| $(1,1)$ | $\text{ReLU}(2)=2$ | $\text{ReLU}(1)=1$ | $2 - 2 = 0$ | 0 ✓ |

Đúng cả 4! Mấu chốt nằm ở dòng cuối: nhờ ReLU **bẻ cong**, nơ-ron $h_2$ chỉ "bật" khi cả hai đầu vào
bằng 1, cho phép tầng ra trừ bớt và kéo $(1,1)$ về 0 — điều một đường thẳng không làm được.

---

## 3. Cài đặt (Implementation)

### 3.1. Các tầng — `ddl/numpy_nn.py`

Ta dựng ba lớp, **cố ý bắt chước PyTorch** (Chương 9–10 sẽ thấy PyTorch tự động hóa đúng những thứ này).
Chương này chỉ viết phần **forward**; phần **backward** (để *học*) thêm ở Chương 7.

```python
# ddl/numpy_nn.py
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
    def forward(self, x):
        self.x = x                  # nhớ input cho backward (Chương 7)
        return x @ self.W + self.b

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask        # giữ phần dương, ép phần âm về 0

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

### 3.2. Chứng minh bằng code — `ch06_forward_pass.py`

Ta đặt tay đúng bộ trọng số ở mục 2.4 rồi chạy forward:

```python
from ddl.numpy_nn import Linear, ReLU, Sequential

net = Sequential([Linear(2, 2), ReLU(), Linear(2, 1)])
net.layers[0].W = np.array([[1., 1.], [1., 1.]]); net.layers[0].b = np.array([0., -1.])
net.layers[2].W = np.array([[1.], [-2.]]);        net.layers[2].b = np.array([0.])
out = net.forward(X)            # -> [0, 1, 1, 0] = XOR
```

---

## 4. Chạy thử (Run it)

```powershell
python ch06_forward_pass.py
```

**Kết quả mong đợi (expected output)** — đây là phép tính cố định nên kết quả khớp chính xác:

```
=== Một đơn vị tuyến tính KHÔNG giải được XOR ===
x1 + x2       = [0. 1. 1. 2.]
XOR mong muốn = [0. 1. 1. 0.]

=== Mạng 2 tầng + ReLU GIẢI ĐƯỢC XOR ===
Hidden sau ReLU:
 [[0. 0.]
 [1. 0.]
 [1. 0.]
 [2. 1.]]
Đầu ra mạng   = [0. 1. 1. 0.]
XOR mong muốn = [0. 1. 1. 0.]
```

Đọc kết quả:

- **Đường thẳng** cho $[0, 1, 1, 2]$ — sai ở $(1,1)$: cần 0 mà ra 2. Không đường thẳng nào sửa được.
- **Mạng 2 tầng** cho đúng $[0, 1, 1, 0]$. Nhìn cột `Hidden`: tầng ẩn đã **biến đổi** không gian đầu vào
  (chú ý riêng điểm $(1,1)$ kích hoạt nơ-ron thứ hai = cột thứ 2 bằng 1), để tầng ra tách được hai lớp.

> **Nhưng** — ta vừa *đặt tay* các trọng số vì đã biết đáp án. Với bài toán thật, không ai biết trước
> $W^{[1]}, W^{[2]}$ phải là gì. Làm sao để mạng **tự tìm ra** chúng từ dữ liệu? Đó chính xác là nội dung
> Chương 7: **lan truyền ngược (backpropagation)**.

---

## 5. Bài tập (Exercises)

1. **Tự kiểm chứng phi tuyến.** Bỏ ReLU đi (dùng `Sequential([Linear(2, 2), Linear(2, 1)])` với cùng
   trọng số đặt tay). Đầu ra còn đúng XOR không? Liên hệ với chứng minh ở mục 2.2.
2. **Đếm tham số.** Mạng XOR có bao nhiêu tham số học được tất cả (đếm số phần tử trong $W^{[1]},
   \mathbf{b}^{[1]}, W^{[2]}, \mathbf{b}^{[2]}$)?
3. **Một bộ trọng số khác.** Kiểm tra bằng tay rằng $h_1 = \text{ReLU}(x_1 - x_2)$, $h_2 = \text{ReLU}(x_2
   - x_1)$, $\hat{y} = h_1 + h_2$ cũng giải được XOR. (Tính cho cả 4 điểm.)
4. **Vì sao không khởi tạo toàn số 0?** Nếu mọi trọng số khởi tạo bằng 0, hãy lập luận vì sao hai nơ-ron
   ẩn sẽ luôn cho cùng một giá trị và mạng không học được gì (gợi ý: chúng nhận cùng gradient).
5. **Đổi hàm kích hoạt.** Thay ReLU bằng sigmoid (đã có trong `ddl.functions`) trong tầng ẩn. Forward
   vẫn chạy chứ? (Chưa cần đúng XOR — chỉ kiểm tra mạng hoạt động với một phi tuyến khác.)

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. Sai. Không có ReLU, hai tầng gập lại thành một phép tuyến tính (mục 2.2) → lại là "đường thẳng" → cho
   $[0,1,1,2]$, không giải được XOR.
2. $W^{[1]}$ có $2\times 2 = 4$, $\mathbf{b}^{[1]}$ có 2, $W^{[2]}$ có $2\times 1 = 2$, $\mathbf{b}^{[2]}$
   có 1 → tổng **9** tham số.
3. $(0,0)$: $h_1=h_2=0 \Rightarrow 0$. $(0,1)$: $h_1=\text{ReLU}(-1)=0, h_2=\text{ReLU}(1)=1 \Rightarrow
   1$. $(1,0)$: $h_1=1,h_2=0 \Rightarrow 1$. $(1,1)$: $h_1=h_2=0 \Rightarrow 0$. Đúng XOR.
4. Khởi tạo 0 → mọi nơ-ron tính ra cùng số, và khi lan truyền ngược chúng nhận **cùng** gradient nên cập
   nhật **y hệt** nhau mãi mãi → không bao giờ tách thành các đặc trưng khác nhau ("đối xứng" không bị
   phá vỡ). Vì vậy ta khởi tạo ngẫu nhiên.
5. Có, forward vẫn chạy: chỉ cần một hàm phi tuyến áp theo từng phần tử là được; sigmoid cũng hợp lệ (dù
   ReLU thường học nhanh hơn cho tầng ẩn).

</details>

---

## 6. Tiếp theo (What's next)

Ta đã thấy một mạng **có thể** biểu diễn quy luật phi tuyến — nhưng mới chỉ bằng cách đặt tay trọng số.
Ở **Chương 7 — Lan truyền ngược bằng tay (Backpropagation by hand)**, ta sẽ trả lời câu hỏi lớn nhất của
cả khóa học: làm sao mạng **tự học** các trọng số đó? Ta sẽ áp **quy tắc chuỗi** xuyên qua từng tầng để
tính gradient, viết phần `backward` cho mỗi lớp, rồi huấn luyện mạng tự giải XOR từ con số 0.
