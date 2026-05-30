# Chương 3 — Toán cần thiết (The math we actually need)

## 0. Chúng ta đang ở đâu (Where we are)

- **Chương 1:** ta có bản đồ quy trình (pipeline) và biết "học" nghĩa là *lặp lại: dự đoán → đo sai số
  → chỉnh tham số*.
- **Chương 2:** ta có công cụ NumPy để viết toán thành code.

Chương này lấp đầy mảnh ghép toán học cho **bước 6 — vòng lặp huấn luyện (training loop)** trên bản đồ.
Cụ thể, ta trả lời câu hỏi cốt lõi: *"Làm sao mô hình biết nên chỉnh tham số theo hướng nào để sai ít
hơn?"* Câu trả lời gồm đúng bốn ý tưởng: **đạo hàm (derivative)**, **gradient**, **gradient descent**,
và **quy tắc chuỗi (chain rule)**. Nắm bốn thứ này là bạn đã hiểu *trái tim* của mọi thuật toán học
trong khóa học. Chương 4 sẽ dùng chúng ngay để huấn luyện mô hình đầu tiên.

> Toán ở đây **rút gọn đúng phần cần dùng** và luôn kèm trực giác + ví dụ số nhỏ. Không cần nền tảng
> giải tích vững — ta xây lại từ đầu.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

Nhớ phép ẩn dụ **ném phi tiêu bịt mắt** ở Chương 1: mỗi lần ném, ta được biết *trượt bao xa* và *về
hướng nào*, rồi chỉnh tay một chút. Bốn khái niệm của chương này chính là phiên bản toán học của câu
"chỉnh tay theo hướng nào":

- **Đạo hàm (derivative)** trả lời: *"nếu tôi nhích đầu vào lên một chút, kết quả thay đổi bao nhiêu và
  theo chiều nào?"* — nói cách khác, nó là **độ dốc (slope)**.
- **Gradient** là đạo hàm cho trường hợp **nhiều tham số cùng lúc**: nó là **mũi tên chỉ hướng dốc lên
  nhanh nhất**. Đi **ngược** mũi tên đó là cách giảm giá trị nhanh nhất.
- **Gradient descent** là việc lặp lại "bước một chút theo hướng ngược gradient" để **trượt xuống đáy**
  (nơi sai số nhỏ nhất).
- **Quy tắc chuỗi (chain rule)** cho ta tính độ dốc xuyên qua **một chuỗi phép biến đổi** — và một mạng
  nơ-ron *chính là* một chuỗi phép biến đổi. Đây là hạt nhân của **lan truyền ngược (backpropagation)**
  ở Chương 7.

---

## 2. Toán học (The math)

### 2.1. Vector, ma trận — ý nghĩa (rất ngắn)

Ở Chương 2 ta đã thao tác với mảng. Về mặt toán: một **vector** $\mathbf{x} = [x_1, \dots, x_D]$ là một
danh sách số (một điểm trong không gian $D$ chiều); một **ma trận** $W$ là một bảng số biến một vector
thành một vector khác (một **phép biến đổi tuyến tính**). Phép toán ta dùng nhiều nhất là **tích vô
hướng (dot product)** và **nhân ma trận** (đã gặp ở Chương 2, mục 2.2). Đó là tất cả những gì cần — phần
còn lại của chương dành cho giải tích.

### 2.2. Đạo hàm (derivative): độ dốc của một hàm

**Định nghĩa.** Đạo hàm của $f$ tại $x$ là giới hạn của "thay đổi đầu ra chia thay đổi đầu vào" khi
bước nhích $h$ tiến về 0:

$$ f'(x) \;=\; \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}. $$

**Trực giác.** $f'(x)$ là **độ dốc của tiếp tuyến** tại điểm $x$. Nếu $f'(x) > 0$, hàm đang **đi lên**
(tăng $x$ thì $f$ tăng); nếu $f'(x) < 0$, hàm đang **đi xuống**; nếu $f'(x) = 0$, ta đang ở chỗ **bằng
phẳng** (có thể là đáy hoặc đỉnh).

**Vài quy tắc ta thực sự dùng** (chỉ cần nhớ bấy nhiêu):

| Hàm $f(x)$ | Đạo hàm $f'(x)$ | Ghi chú |
|------------|-----------------|---------|
| hằng số $c$ | $0$ | đường nằm ngang, không dốc |
| $x$ | $1$ | |
| $x^n$ | $n\,x^{n-1}$ | "quy tắc lũy thừa (power rule)"; ví dụ $x^2 \to 2x$ |
| $c\cdot f(x)$ | $c\cdot f'(x)$ | hằng số đi ra ngoài |
| $f(x) + g(x)$ | $f'(x) + g'(x)$ | đạo hàm của tổng = tổng đạo hàm |

**Ví dụ số.** Với $f(x) = x^2$ thì $f'(x) = 2x$. Tại $x = 3$, độ dốc là $2\cdot 3 = 6$. Ở mục "Chạy thử"
ta sẽ kiểm chứng con số 6 này bằng cách nhích $x$ một lượng nhỏ.

### 2.3. Đạo hàm riêng & gradient (nhiều biến)

Hàm mất mát của ta sẽ phụ thuộc **nhiều tham số** cùng lúc, ví dụ $g(w_0, w_1)$. **Đạo hàm riêng
(partial derivative)** theo $w_0$, viết $\dfrac{\partial g}{\partial w_0}$, nghĩa là: *coi mọi biến khác
như hằng số*, rồi lấy đạo hàm theo riêng $w_0$.

**Gradient** là **vector gom tất cả đạo hàm riêng** lại:

$$ \nabla g = \left[\frac{\partial g}{\partial w_0},\; \frac{\partial g}{\partial w_1}\right]. $$

**Trực giác.** Hãy hình dung $g$ như độ cao của một ngọn đồi tại tọa độ $(w_0, w_1)$. Gradient $\nabla g$
là **mũi tên chỉ hướng lên dốc nhanh nhất** tại chỗ bạn đang đứng; độ dài mũi tên cho biết dốc đến đâu.

**Ví dụ số.** Với $g(w_0, w_1) = w_0^2 + w_1^2$:

$$ \frac{\partial g}{\partial w_0} = 2w_0, \quad \frac{\partial g}{\partial w_1} = 2w_1, \quad
   \nabla g = [\,2w_0,\; 2w_1\,]. $$

Tại $w = [1, 2]$: $\nabla g = [2, 4]$. Đáy của "cái bát" $g$ nằm ở gốc $[0,0]$, và để đi về đáy ta nên
đi **ngược** hướng $[2, 4]$ — tức về phía gốc. Đó chính là ý tưởng tiếp theo.

### 2.4. Huấn luyện = tối thiểu hóa một hàm — và vì sao không "giải thẳng" được

Hãy phát biểu lại mục tiêu cho thật rõ: **huấn luyện một mô hình = tìm bộ tham số $\theta$ làm cho hàm
mất mát $J(\theta)$ nhỏ nhất.** Nói cách khác, đó là một bài toán **tối ưu hóa (optimization)**: tìm
**đáy** của một hàm.

Bạn có thể hỏi: *"Giải tích chẳng phải đã dạy cách tìm cực tiểu rồi sao — cho đạo hàm bằng 0 rồi giải
ra?"* Đúng, với hàm **đơn giản** thì làm được. Ví dụ $f(x) = x^2$: giải $f'(x) = 2x = 0$ ra ngay
$x = 0$. Đó là lời giải **dạng đóng (closed-form)** — một công thức cho thẳng đáp án. (Thật ra hồi quy
tuyến tính ở Chương 4 *cũng* có một công thức đóng như vậy.)

Nhưng với một mạng nơ-ron sâu, cách "giải thẳng" **sụp đổ**:

1. **Quá nhiều biến.** $J$ phụ thuộc hàng triệu (đến hàng tỉ) tham số. "Cho gradient bằng 0" trở thành
   một hệ hàng triệu phương trình phi tuyến chằng chịt — **không có công thức** để giải.
2. **Quá đắt.** Ngay cả những trường hợp giải được trên giấy cũng đòi hỏi nghịch đảo (invert) những ma
   trận khổng lồ — tốn kém đến mức bất khả thi.
3. **Không có toàn cảnh.** Ta thường thậm chí không viết được công thức gọn cho $J$ trên toàn bộ dữ liệu.

Vậy ta cần một phương pháp **khác hẳn**: thay vì giải đúng một phát (bất khả thi), hãy **cải thiện dần
từng chút một** (dễ). Phương pháp đó chỉ cần làm được hai việc tại điểm hiện tại: (a) tính giá trị $J$,
và (b) tính **gradient** $\nabla J$. Đó chính là **gradient descent**. Nó đánh đổi "giải đúng một lần
(không thể)" lấy "cải thiện một chút, lặp thật nhiều lần (làm được)".

> **Sao không thử ngẫu nhiên (random search)?** Trong không gian hàng triệu chiều, dò ngẫu nhiên gần
> như **không bao giờ** tình cờ tốt lên (đây là "lời nguyền số chiều" — curse of dimensionality).
> Gradient thì *tặng không* cho ta một hướng **chắc chắn** làm mất mát giảm. Đó là lý do nó mạnh.

### 2.5. Vì sao gradient lại đúng là hướng "dốc nhất"? (phần khó nhất — đi từ từ)

Trong **1 chiều**, mọi thứ hiển nhiên: đạo hàm dương nghĩa là dốc lên, âm nghĩa là dốc xuống; muốn giảm
thì đi ngược dấu đạo hàm. Nhưng trong **nhiều chiều**, từ một điểm bạn có thể bước theo **vô số hướng**.
Câu hỏi sống còn: bước theo hướng nào thì hàm giảm **nhanh nhất**?

**Bước 1 — "Độ dốc theo một hướng" là gì?** Chọn một hướng = một **vector đơn vị (unit vector)**
$\mathbf{d}$ (độ dài đúng bằng 1). Nếu nhích một đoạn nhỏ theo $\mathbf{d}$, hàm thay đổi với một tỉ lệ
gọi là **đạo hàm theo hướng (directional derivative)**. Nó có một công thức rất gọn — bằng **tích vô
hướng** của gradient với $\mathbf{d}$:

$$ \text{độ dốc theo hướng } \mathbf{d} \;=\; \nabla f \cdot \mathbf{d}. $$

*Vì sao là tích vô hướng?* Mỗi thành phần của $\mathbf{d}$ cho biết ta đi bao nhiêu theo mỗi trục; nhân
với độ dốc theo trục đó (chính là đạo hàm riêng $\partial f/\partial w_i$) rồi cộng tất cả lại — đó đúng
là định nghĩa của tích vô hướng $\sum_i (\partial f/\partial w_i)\, d_i$.

**Bước 2 — Hình học của tích vô hướng.** Nhớ lại công thức: $\nabla f \cdot \mathbf{d} = |\nabla f|\,
|\mathbf{d}|\cos\alpha$, với $\alpha$ là góc giữa hai vector. Vì $\mathbf{d}$ là vector đơn vị
($|\mathbf{d}| = 1$):

$$ \text{độ dốc theo hướng } \mathbf{d} \;=\; |\nabla f|\,\cos\alpha. $$

**Bước 3 — Tìm hướng tốt nhất.** $\cos\alpha$ chỉ chạy trong khoảng $[-1, 1]$, nên độ dốc theo hướng chỉ
chạy từ $-|\nabla f|$ đến $+|\nabla f|$. Hai đầu mút này chính là câu trả lời:

- Dốc **lên** mạnh nhất ($+|\nabla f|$) khi $\alpha = 0^\circ$, tức $\mathbf{d}$ **cùng hướng** $\nabla
  f$. ⟹ **gradient là hướng dốc lên nhanh nhất (steepest ascent).**
- Dốc **xuống** mạnh nhất ($-|\nabla f|$) khi $\alpha = 180^\circ$, tức $\mathbf{d} = -\nabla f/|\nabla
  f|$. ⟹ **ngược gradient là hướng dốc xuống nhanh nhất (steepest descent).**

Đây chính là lý do **toán học** vì sao gradient descent đi theo $-\nabla f$: trong **mọi** hướng có thể
bước, đó là hướng làm hàm giảm nhiều nhất cho mỗi bước.

**Phép ẩn dụ ngọn đồi trong sương mù.** Bạn đứng trên sườn đồi, sương dày đặc không thấy gì, nhưng chân
vẫn cảm nhận được độ dốc theo mỗi hướng định bước. Gradient giống một **chiếc kim la bàn luôn chỉ thẳng
lên phía dốc nhất**. Bước theo bất kỳ hướng nào, độ dốc bạn cảm thấy = mức độ hướng đó "ăn khớp" với
chiếc kim (chính là $\cos\alpha$). Muốn **xuống** nhanh nhất ư? Chỉ việc quay lưng lại và đi **ngược
chiều chiếc kim**. Mục "Chạy thử" sẽ cho bạn thấy bằng số: trong các hướng đem ra thử, hướng $-\nabla f$
cho độ dốc âm nhất, đúng bằng $-|\nabla f|$.

> **Cục bộ, không phải toàn cục (local vs global).** Gradient chỉ ra hướng tốt nhất **ngay tại chỗ đang
> đứng**. Lặp lại nhiều bước, ta trượt dần xuống một **đáy** (nơi gradient $= \mathbf{0}$, mặt bằng
> phẳng). Với hàm hình "cái bát" như $x^2$ chỉ có **một** đáy nên chắc chắn về đúng đó. Mặt mất mát của
> mạng sâu thì gồ ghề, nhiều "thung lũng" (cực tiểu cục bộ — local minima), nhưng thực nghiệm cho thấy
> gradient descent vẫn tìm được những đáy **đủ tốt** — ta sẽ gặp lại chủ đề này ở các chương sau.

### 2.6. Quy tắc cập nhật & tốc độ học (the update rule & learning rate)

Gộp lại: đứng tại $\theta$, đi một bước nhỏ theo hướng dốc xuống nhanh nhất ($-\nabla J$), rồi lặp lại:

$$ \boxed{\;\theta \;\leftarrow\; \theta \;-\; \eta\, \nabla_\theta J\;} $$

$J$ là hàm cần giảm (sau này là **mất mát**), còn $\eta$ (eta) là **tốc độ học (learning rate)** — độ dài
mỗi bước. Nó nhỏ quan trọng nhưng ảnh hưởng rất lớn:

- $\eta$ **quá nhỏ** → mỗi bước tí xíu → học rất chậm.
- $\eta$ **quá lớn** → bước quá đà, có thể **nhảy vọt qua đáy** rồi văng ra xa, không hội tụ (diverge).

**Ví dụ số (tính tay).** Giảm $f(x) = x^2$, bắt đầu $x = 4$, $\eta = 0.1$. Vì $f'(x) = 2x$, cập nhật là
$x \leftarrow x - 0.1\cdot(2x) = x - 0.2x = 0.8x$. Vậy mỗi bước $x$ co lại còn 80%:

$$ 4 \to 3.2 \to 2.56 \to 2.048 \to \dots \to 0. $$

$x$ tiến dần về 0 — đúng đáy của $f$. Đây là **toàn bộ** cơ chế "học": lặp lại quy tắc cập nhật cho đến
khi mất mát đủ nhỏ.

### 2.7. Vì sao deep learning *bắt buộc* cần gradient descent

Giờ ghép các mảnh lại để thấy vì sao đây là **động cơ (engine)** của cả lĩnh vực:

- Một mô hình deep learning là **nhiều tầng xếp chồng** nên hàm mất mát $J(\theta)$ cực kỳ phức tạp,
  hàng triệu chiều, **không có công thức** tìm đáy (mục 2.4).
- Nhưng có **đúng hai thứ** ta tính được hiệu quả: giá trị mất mát (qua **lượt tính xuôi — forward
  pass**) và gradient của nó (qua **lan truyền ngược — backpropagation**, Chương 7). Gradient descent
  được xây *chính xác* từ hai nguyên liệu này — nên nó khớp một cách hoàn hảo.
- Nó **co giãn (scales)** tốt: với dữ liệu khổng lồ, ta tính gradient trên từng lô nhỏ (mini-batch) —
  gọi là **stochastic gradient descent**, Chương 8; với mô hình hàng tỉ tham số, nó vẫn chạy được.

Gần như **mọi** mô hình deep learning bạn từng nghe tên đều được huấn luyện bằng gradient descent (hoặc
một biến thể tinh chỉnh như **Adam**, Chương 10). Hiểu vững gradient descent = hiểu *cách mọi mô hình
học*. Đó là lý do ta đầu tư nhiều công đến vậy cho nó.

### 2.8. Quy tắc chuỗi (chain rule): hạt nhân của backprop

Khi một đại lượng phụ thuộc đầu vào **qua nhiều tầng**, độ dốc tổng = **tích các độ dốc** từng tầng.
Nếu $y = f(u)$ và $u = g(x)$ thì:

$$ \frac{dy}{dx} \;=\; \frac{dy}{du}\cdot\frac{du}{dx}. $$

**Trực giác.** Hãy nghĩ về tốc độ: nếu $u$ thay đổi nhanh gấp 3 lần $x$, và $y$ thay đổi nhanh gấp 2 lần
$u$, thì $y$ thay đổi nhanh gấp $2\times 3 = 6$ lần $x$. Các "tỉ lệ thay đổi" **nhân nhau** dọc theo
chuỗi.

**Ví dụ số.** $y = (3x + 1)^2$. Đặt $u = 3x + 1$ nên $y = u^2$:

$$ \frac{dy}{du} = 2u, \qquad \frac{du}{dx} = 3
   \;\;\Rightarrow\;\; \frac{dy}{dx} = 2u\cdot 3 = 6(3x+1). $$

Tại $x = 1$: $u = 4$, $\dfrac{dy}{dx} = 6\cdot 4 = 24$.

**Vì sao điều này là cốt lõi của deep learning?** Một mạng nơ-ron là một **chuỗi hàm lồng nhau**: đầu
vào → tầng 1 → kích hoạt → tầng 2 → … → mất mát. Muốn biết "nếu nhích một trọng số ở **tầng đầu** thì
**mất mát** đổi bao nhiêu", ta **nhân các độ dốc cục bộ** dọc theo chuỗi — đi từ mất mát ngược về trọng
số đó. Quá trình đi ngược này tên là **lan truyền ngược (backpropagation)**, và Chương 7 sẽ làm nó tường
tận từng bước.

---

## 3. Cài đặt (Implementation)

Để *thấy* bốn ý tưởng trên là thật, ta kiểm chứng chúng bằng số. Ý tưởng then chốt: có thể **ước lượng
đạo hàm** mà không cần giải tích, chỉ bằng định nghĩa — nhích đầu vào một lượng nhỏ $h$ và đo thay đổi:

```python
def numerical_derivative(f, x, h=1e-5):
    # Đạo hàm = độ dốc khi nhích x một lượng rất nhỏ h (central difference).
    return (f(x + h) - f(x - h)) / (2 * h)
```

Với hàm nhiều biến, ta nhích **lần lượt từng biến** để lấy từng đạo hàm riêng, gom lại thành gradient:

```python
def numerical_gradient(func, w, h=1e-5):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy();  w_plus[i]  += h   # chỉ nhích biến thứ i
        w_minus = w.copy(); w_minus[i] -= h
        grad[i] = (func(w_plus) - func(w_minus)) / (2 * h)   # đạo hàm riêng theo biến i
    return grad
```

Và một vòng lặp **gradient descent** đúng như quy tắc cập nhật ở mục 2.6:

```python
x = 4.0
lr = 0.1
for step in range(6):
    grad = 2 * x            # f'(x) = 2x  cho f(x)=x^2
    x = x - lr * grad       # theta <- theta - lr * grad
```

Cuối cùng, để *tận mắt* thấy "gradient là hướng dốc nhất" (mục 2.5), ta đo độ dốc theo hướng — chính là
$\nabla f \cdot \mathbf{d}$ — cho nhiều hướng đơn vị $\mathbf{d}$, rồi xác nhận hướng $-\nabla f$ cho độ
dốc **âm nhất** (bằng $-|\nabla f|$):

```python
grad = np.array([2.0, 4.0])            # gradient cua g tai w=[1,2]
grad_norm = np.linalg.norm(grad)       # |grad| = sqrt(20)
d_neg = -grad / grad_norm              # huong nguoc gradient (da chuan hoa ve do dai 1)
print(grad @ d_neg)                     # = -|grad|, am nhat trong moi huong don vi
```

Toàn bộ phần minh họa (gồm cả kiểm chứng quy tắc chuỗi và phép so sánh hướng dốc) nằm trong
`ch03_calculus_demo.py`.

---

## 4. Chạy thử (Run it)

```powershell
python ch03_calculus_demo.py
```

**Kết quả mong đợi (expected output)** — vài chữ số cuối của phần số thực có thể **hơi khác** trên máy
bạn (đạo hàm bằng số là *xấp xỉ*), nhưng phải rất sát giá trị giải tích:

```
=== 1) Đạo hàm của f(x)=x^2 tại x=3 ===
  numerical : 6.000000
  2x        : 6.000000
=== 2) Gradient của g(w)=w0^2+w1^2 tại w=[1,2] ===
  numerical : [2. 4.]
  [2w0,2w1] : [2. 4.]
=== 3) Gradient descent trên f(x)=x^2 (x0=4, lr=0.1) ===
  step 1: x = 3.2000, f(x) = 10.2400
  step 2: x = 2.5600, f(x) = 6.5536
  step 3: x = 2.0480, f(x) = 4.1943
  step 4: x = 1.6384, f(x) = 2.6844
  step 5: x = 1.3107, f(x) = 1.7180
  step 6: x = 1.0486, f(x) = 1.0995
=== 4) Quy tắc chuỗi cho y=(3x+1)^2 tại x=1 ===
  numerical : 24.000000
  6*(3x+1)  : 24.000000
=== 5) Gradient là hướng dốc nhất (steepest direction) ===
  độ dốc theo -trục w0   : -2.0000
  độ dốc theo -trục w1   : -4.0000
  độ dốc theo chéo 45 độ : -4.2426
  độ dốc theo -gradient  : -4.4721
  => âm nhất = -|grad|    : -4.4721  (chỉ đạt được khi đi ngược gradient)
```

Bốn điều đáng chú ý khi đối chiếu với phần toán:

- **Phần 1 & 4:** đạo hàm tính **bằng số** (chỉ nhích đầu vào) khớp với đạo hàm **giải tích** (6 và 24).
  Định nghĩa đạo hàm là thật, không phải phép thuật.
- **Phần 3:** $x$ giảm dần $4 \to 3.2 \to 2.56 \to \dots$, đúng quy luật "co còn 80%" ta tính tay, và
  $f(x)$ ngày càng nhỏ — mô hình đang **học** cách đạt đáy.
- **Phần 5:** trong mọi hướng đem ra thử, hướng **ngược gradient** cho độ dốc âm nhất ($-|\nabla g|
  \approx -4.4721$) — đây là bằng chứng bằng số cho khẳng định "gradient là hướng dốc nhất" ở mục 2.5.

---

## 5. Bài tập (Exercises)

1. **Đạo hàm cơ bản.** Tính $f'(x)$ cho: (a) $f(x) = x^2$; (b) $f(x) = 3x + 1$; (c) $f(x) = x^2 + 2x$;
   (d) $f(x) = 5$.
2. **Đạo hàm riêng.** Cho $g(w_0, w_1) = 3w_0 + w_1^2$. Tính $\dfrac{\partial g}{\partial w_0}$ và
   $\dfrac{\partial g}{\partial w_1}$, rồi viết gradient $\nabla g$ tại $w = [4, 2]$.
3. **Một bước gradient descent (tính tay).** Với $f(x) = x^2$, đang ở $x = 5$, $\eta = 0.1$. Tính $x$
   **sau một bước** cập nhật. Sau bước đó, $f(x)$ tăng hay giảm?
4. **Quy tắc chuỗi.** Cho $y = (2x - 1)^3$. Dùng phép đặt $u = 2x - 1$ để tính $\dfrac{dy}{dx}$, rồi tính
   giá trị tại $x = 1$.
5. **(Suy nghĩ)** Vì sao quy tắc cập nhật dùng **trừ** ($\theta \leftarrow \theta - \eta\nabla J$) chứ
   không phải cộng? Nếu đổi thành **cộng**, $x$ trong Phần 3 sẽ đi về đâu?
6. **Hướng dốc nhất.** Xét $g(w) = w_0^2 + w_1^2$ tại $w = [3, 4]$, ở đó gradient là $\nabla g = [6, 8]$.
   (a) Tính độ dài $|\nabla g|$. (b) Viết **vector đơn vị** chỉ hướng **giảm** $g$ nhanh nhất. (c) Độ dốc
   theo hướng đó bằng bao nhiêu?

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. (a) $2x$; (b) $3$; (c) $2x + 2$; (d) $0$.
2. $\partial g/\partial w_0 = 3$ (coi $w_1$ là hằng); $\partial g/\partial w_1 = 2w_1$. Tại $[4,2]$:
   $\nabla g = [3, 4]$.
3. $x \leftarrow 5 - 0.1\cdot(2\cdot 5) = 5 - 1 = 4$. $f$ giảm từ $25$ xuống $16$.
4. $\dfrac{dy}{dx} = 3u^2\cdot 2 = 6(2x-1)^2$. Tại $x=1$: $6\cdot 1^2 = 6$.
5. Vì gradient chỉ hướng **lên dốc**; muốn **giảm** hàm ta đi ngược lại, nên dùng dấu trừ. Nếu dùng dấu
   cộng, $x$ sẽ **leo lên** và tiến ra vô cực (phát tán) — sai số càng lúc càng lớn.
6. (a) $|\nabla g| = \sqrt{6^2 + 8^2} = \sqrt{100} = 10$. (b) hướng giảm nhanh nhất là $-\nabla g/|\nabla
   g| = [-0.6,\, -0.8]$. (c) độ dốc theo hướng đó là $-|\nabla g| = -10$ (âm nhất có thể).

</details>

---

## 6. Tiếp theo (What's next)

Bạn đã có đủ bốn mảnh ghép toán: đạo hàm, gradient, gradient descent, quy tắc chuỗi. Ở **Chương 4 — Hồi
quy tuyến tính từ số 0 (Linear regression from scratch)**, ta sẽ ghép tất cả lại để xây **mô hình biết
tự học** đầu tiên: chọn một đường thẳng, đo sai số bằng **hàm mất mát (loss)**, rồi dùng **gradient
descent** để mô hình tự tìm đường thẳng tốt nhất — toàn bộ viết tay bằng NumPy.
