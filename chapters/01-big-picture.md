# Chương 1 — Bức tranh tổng thể (The big picture)

## 0. Chúng ta đang ở đâu (Where we are)

Đây là chương đầu tiên, nên chưa có gì để nhắc lại. Mục tiêu của chương này **không phải** để bạn viết
được một mạng nơ-ron — mà để bạn có một **tấm bản đồ (map)** trong đầu: huấn luyện (train) một mô hình
deep learning gồm những bước nào, và **vì sao** lại cần từng bước đó.

Tại sao điều này quan trọng đến vậy? Vì nếu ta lao thẳng vào "thu thập dữ liệu", rồi "tiền xử lý
(preprocessing)", rồi "xây mô hình"... mà bạn chưa thấy bức tranh lớn, thì mỗi bước sẽ giống như một
mệnh lệnh khó hiểu: *"làm cái này đi"* mà không rõ để làm gì. Có bản đồ rồi, mỗi chương sau sẽ là việc
phóng to (zoom in) vào **một** ô trên bản đồ — và bạn luôn biết ô đó nằm ở đâu trong toàn cảnh.

> Quy ước nhỏ trong khóa học: phần giải thích/lý thuyết viết bằng tiếng Việt, kèm **thuật ngữ tiếng
> Anh trong ngoặc** ở lần đầu xuất hiện. Code và tên biến (identifiers) viết bằng tiếng Anh, nhưng
> phần chú thích (comments) thì viết bằng tiếng Việt để bạn dễ hiểu.

---

## 1. Trực giác & ý tưởng (Intuition & idea)

### 1.1. Lập trình truyền thống so với học máy (Traditional programming vs machine learning)

Hãy bắt đầu từ một câu hỏi rất cụ thể: làm sao để máy tính **nhận ra chữ số viết tay**? Ví dụ, nhìn vào
một ảnh và nói "đây là số 7".

Trong **lập trình truyền thống (traditional programming)**, bạn — người lập trình — phải tự nghĩ ra
**luật (rules)**:

> "Nếu có một nét ngang ở trên và một nét chéo đi xuống thì đó là số 7..."

Cách này thất bại nhanh chóng: chữ viết tay của mỗi người mỗi khác, có hàng nghìn ngoại lệ, và bạn
không thể viết hết luật cho mọi kiểu nét. Đây là sơ đồ của lập trình truyền thống:

```
Luật (do người viết)  +  Dữ liệu đầu vào   →   [ Chương trình ]   →   Đáp án
(rules)                  (input data)                                 (output)
```

**Học máy (machine learning)** lật ngược sơ đồ đó lại. Thay vì viết luật, ta đưa cho máy **rất nhiều ví
dụ** (ảnh chữ số kèm đáp án đúng) và để máy **tự tìm ra luật**:

```
Dữ liệu đầu vào  +  Đáp án đúng   →   [ Quá trình học (training) ]   →   Luật (model)
(input data)        (labels)                                            (the model)
```

Cái "luật" mà máy tìm ra chính là **mô hình (model)**. Sau khi học xong, ta dùng mô hình này như một
chương trình bình thường: đưa ảnh mới vào, nó trả về đáp án.

### 1.2. Mô hình thực chất là một hàm số (A model is just a function)

Đừng để chữ "mô hình" làm bạn sợ. Về bản chất, một mô hình chỉ là một **hàm số (function)**: nhận đầu
vào $x$ (ví dụ: các điểm ảnh của bức ảnh) và trả ra dự đoán $\hat{y}$ (ví dụ: "số 7").

$$ \hat{y} = f_\theta(x) $$

Trong đó:

- $x$ là **đầu vào (input)** — dữ liệu ta đưa vào.
- $\hat{y}$ (đọc là "y-hat") là **dự đoán (prediction)** của mô hình.
- $f$ là **dạng (form)** của hàm — ta tự chọn (đường thẳng? mạng nơ-ron?).
- $\theta$ (đọc là "theta") là **các tham số (parameters)** — những con số bên trong hàm mà máy sẽ
  điều chỉnh dần trong lúc học. Đây chính là thứ được "học".

Cả khóa học này, xét cho cùng, chỉ xoay quanh một câu hỏi: **làm sao tìm được bộ tham số $\theta$ tốt
nhất**, sao cho dự đoán $\hat{y}$ gần với đáp án đúng $y$ nhất có thể.

### 1.3. "Học" nghĩa là gì? — phép ẩn dụ ném phi tiêu (What does "learning" mean?)

Hình dung bạn tập ném phi tiêu (darts) khi bịt mắt. Mỗi lần ném, có người nói cho bạn biết bạn trượt
*bao xa* và *về hướng nào*. Bạn chỉnh tay một chút theo hướng đó, rồi ném lại. Lặp lại nhiều lần, bạn
ngày càng gần tâm.

Máy học theo đúng kiểu đó:

1. Mô hình **dự đoán** (ném phi tiêu).
2. Ta đo xem dự đoán **sai bao nhiêu** — con số này gọi là **mất mát (loss)**.
3. Ta tính xem nên chỉnh các tham số $\theta$ **theo hướng nào** để lần sau sai ít hơn — hướng này gọi
   là **gradient**.
4. Ta chỉnh $\theta$ một chút theo hướng đó, rồi lặp lại.

Quá trình lặp đi lặp lại "dự đoán → đo sai số → chỉnh" này chính là **huấn luyện (training)**, và thuật
toán chỉnh tham số tên là **gradient descent** (tạm dịch: "đi xuống theo gradient"). Toàn bộ phần toán
của khóa học là để làm cho 4 bước trên thật chính xác — nhưng trực giác thì đơn giản đúng như ném phi
tiêu.

---

## 2. Toán học (The math) — bản đồ toàn bộ quy trình

Chương này nhẹ về công thức; "toán" ở đây là **bản đồ quy trình (the pipeline)**. Hãy ghi nhớ sơ đồ
sau — mọi chương về sau đều là phóng to một ô trong đây:

```
        ┌─────────────────────── GIAI ĐOẠN HUẤN LUYỆN (TRAINING) ───────────────────────┐
        │                                                                                │
 (1) Thu thập   (2) Chia      (3) Tiền xử lý    (4) Xây        (5) Chọn hàm              │
 dữ liệu    →   dữ liệu   →   dữ liệu      →    mô hình   →    mất mát        ─┐         │
 (collect)      (split)       (preprocess)      (build model)  (loss)          │         │
                                                                               ▼         │
                                                              ┌──── VÒNG LẶP HUẤN LUYỆN ─┐│
                                                              │ (6) forward → loss →     ││
                                                              │     backward → update    ││  ← lặp nhiều lần
                                                              └──────────┬───────────────┘│
        │                                                               ▼                 │
        │                                                    (7) Đánh giá (evaluate)       │
        └───────────────────────────────────────────────────────────┬────────────────────┘
                                                                      ▼
                                              (8) Lưu mô hình (save)  →  (9) Suy luận (inference)
                                                                          dùng cho dữ liệu mới
```

Bây giờ là phần quan trọng nhất của cả chương: **vì sao cần từng bước**. Hãy đọc kỹ bảng này — nó là lý
do tồn tại của 13 chương còn lại.

| Bước | Tên (English) | Nó làm gì | **Tại sao cần nó** | Học ở chương |
|------|---------------|-----------|--------------------|--------------|
| 1 | Thu thập dữ liệu (collect data) | Lấy các ví dụ kèm đáp án đúng | Máy học **từ ví dụ**. Không có dữ liệu thì không có gì để học. | 8 |
| 2 | Chia dữ liệu (split data) | Tách thành tập huấn luyện / kiểm định / kiểm tra (train / validation / test) | Để biết mô hình có thực sự **học được quy luật** hay chỉ **học thuộc lòng** dữ liệu cũ. Phải kiểm tra trên dữ liệu nó **chưa từng thấy**. | 8 |
| 3 | Tiền xử lý (preprocessing) | Đưa dữ liệu về dạng số, cùng thang đo (ví dụ chuẩn hóa về quanh 0) | Mô hình chỉ hiểu **số**, và học **nhanh & ổn định hơn nhiều** khi các đặc trưng có cùng thang đo. | 4, 8 |
| 4 | Xây mô hình (build model) | Chọn dạng hàm $f_\theta$ (đường thẳng, MLP, CNN...) | Đây là "bộ khung" có các tham số $\theta$ để máy điều chỉnh. Dạng hàm càng phù hợp bài toán, học càng tốt. | 4–7, 10, 13 |
| 5 | Chọn hàm mất mát (loss function) | Một con số đo "dự đoán sai bao nhiêu" | Nếu không **đo** được độ sai thì không biết đang tốt lên hay tệ đi. Loss là "tấm gương" để mô hình soi. | 4, 5 |
| 6 | Vòng lặp huấn luyện (training loop) | forward → loss → backward → cập nhật tham số, lặp lại | Đây là chỗ việc **học thật sự** diễn ra: lặp lại để giảm dần loss (đúng như ném phi tiêu). | 4, 7, 8, 11 |
| 7 | Đánh giá (evaluate) | Đo độ chính xác (accuracy) trên tập chưa từng thấy | Để **chấm điểm** mô hình một cách trung thực và phát hiện học tủ (overfitting). | 8, 11 |
| 8 | Lưu mô hình (save) | Ghi các tham số đã học ra đĩa | Để **không phải huấn luyện lại** mỗi lần dùng. Train một lần, dùng nhiều lần. | 11 |
| 9 | Suy luận (inference) | Dùng mô hình đã học cho **dữ liệu mới** | Đây là **mục đích cuối cùng**: đưa một ảnh mới vào và nhận về dự đoán. | 11, 14 |

Một vài điều cần thấm ngay từ bây giờ:

- **Huấn luyện (training)** và **suy luận (inference)** là hai chế độ khác nhau. Lúc train, mô hình
  *đang học* (tham số thay đổi). Lúc inference, mô hình *đã học xong* (tham số đứng yên), ta chỉ lấy
  dự đoán. Sự phân biệt này sẽ quay lại nhiều lần (ví dụ `model.train()` vs `model.eval()` ở Chương 11).
- **Vì sao phải chia dữ liệu (bước 2)?** Tưởng tượng một học sinh được phát trước đề thi kèm đáp án.
  Em đó được 10 điểm, nhưng đó là vì **học thuộc**, không phải vì **hiểu bài**. Muốn biết em có thực sự
  hiểu, phải cho làm một đề **mới**. Tập kiểm tra (test set) chính là "đề mới" đó. Hiện tượng "học
  thuộc mà không hiểu" gọi là **overfitting**, và ta sẽ gặp nó tận mắt ở Chương 8.

Đừng cố nhớ hết bảng này. Chỉ cần biết: **bản đồ tồn tại, và mỗi chương tới là một ô trên bản đồ.** Mỗi
chương sẽ mở đầu bằng mục "Chúng ta đang ở đâu" để chỉ đúng ô đó.

---

## 3. Cài đặt (Implementation) — chuẩn bị môi trường

Chương này chưa code mô hình; việc của ta là **dựng môi trường (environment)** để các chương sau chạy
được, và chạy một đoạn kiểm tra nhỏ cho chắc.

Ta sẽ dùng:

- **Python 3.10+** — ngôn ngữ chính.
- **NumPy** — thư viện tính toán trên mảng số (arrays); dùng cho phần "làm tay từ số 0".
- **PyTorch** (`torch`) + **torchvision** — thư viện deep learning, dùng từ Chương 9; torchvision còn
  giúp tải bộ dữ liệu MNIST.
- **Matplotlib** — để vẽ hình.

### 3.1. Tạo thư mục dự án và môi trường ảo (virtual environment)

**Môi trường ảo (virtual environment)** là một "hộp" Python riêng cho dự án này, để các thư viện ta cài
không lẫn với phần còn lại của máy. Đây là thói quen tốt, nên làm ngay.

Mở terminal (trên Windows: **PowerShell**) và chạy các lệnh ở mục "Chạy thử" bên dưới.

### 3.2. Đoạn mã kiểm tra môi trường (`hello.py`)

Tạo một file tên `hello.py` ở thư mục gốc của dự án với nội dung sau. Mục đích của nó chỉ là in ra
phiên bản các thư viện và cho bạn "nếm thử" một tensor đầu tiên.

```python
# hello.py
# Đoạn mã nhỏ để xác nhận môi trường (environment) đã sẵn sàng.
# (Chưa có machine learning gì ở đây — chỉ kiểm tra các thư viện đã cài đúng.)

import sys
import numpy as np
import torch

# 1) In ra phiên bản (version) các thư viện để chắc chắn import thành công.
print("Python :", sys.version.split()[0])
print("NumPy  :", np.__version__)
print("PyTorch:", torch.__version__)

# 2) Kiểm tra theo kiểu device-agnostic (Contract §3): dùng GPU CUDA nếu PyTorch
#    nhìn thấy một cái, ngược lại thì dùng CPU. Mọi chương sau đều dựa trên ý tưởng này.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

# 3) "Nếm thử" 2 dòng cho biết sắp tới có gì. Một `tensor` chỉ là một mảng số
#    (giống mảng NumPy) mà PyTorch có thể tính toán trên đó — và về sau, học (learn) từ nó.
x = torch.arange(5.0)            # -> tensor([0., 1., 2., 3., 4.])
print("x         =", x)
print("x squared =", x ** 2)     # bình phương từng phần tử (element-wise) -> tensor([0., 1., 4., 9., 16.])
```

> Lưu ý: ở đây ta chưa "học" gì cả. `hello.py` chỉ xác nhận công cụ đã sẵn sàng. Toàn bộ ý nghĩa của
> `tensor`, `device`, `autograd`... sẽ được giải thích kỹ ở Chương 9. Bây giờ chỉ cần nó **chạy được**.

---

## 4. Chạy thử (Run it)

Mở **PowerShell** tại thư mục dự án (`learn-deep-learning`) và chạy lần lượt. (Bản lệnh cho macOS/Linux
ở ngay bên dưới.)

**Trên Windows (PowerShell):**

```powershell
# 1) Tạo môi trường ảo tên .venv
python -m venv .venv

# 2) Kích hoạt (activate) nó — dấu nhắc sẽ có tiền tố (.venv)
.\.venv\Scripts\Activate.ps1

# 3) Cài các thư viện cần dùng cho cả khóa học (bản CPU, chạy mọi máy)
pip install numpy torch torchvision matplotlib

# 4) Chạy đoạn kiểm tra
python hello.py
```

**Trên macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch torchvision matplotlib
python hello.py
```

**Kết quả mong đợi (expected output)** — các con số phiên bản và máy của bạn có thể **hơi khác** (your
numbers may vary a little), điều đó hoàn toàn bình thường:

```
Python : 3.11.5
NumPy  : 1.26.4
PyTorch: 2.2.1
Device : cpu
x         = tensor([0., 1., 2., 3., 4.])
x squared = tensor([ 0.,  1.,  4.,  9., 16.])
```

Nếu máy bạn có GPU NVIDIA và đã cài đúng bản PyTorch hỗ trợ CUDA, dòng `Device` sẽ in `cuda` thay vì
`cpu` — code của ta viết theo kiểu **device-agnostic** nên chạy được cả hai mà không cần đổi gì.

**Mẹo xử lý sự cố (troubleshooting):**

- `python: command not found` → thử `python3` thay cho `python`. Hãy chắc Python 3.10+ đã được cài.
- Trên Windows, nếu dòng `Activate.ps1` báo lỗi về **execution policy**, chạy một lần lệnh sau rồi thử
  lại bước kích hoạt:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
- `pip install torch` chậm hoặc lỗi → vào trang chính thức https://pytorch.org/get-started/locally/ để
  lấy đúng lệnh cài cho hệ máy của bạn (đặc biệt nếu muốn bản GPU/CUDA).

> Từ giờ, **mọi lệnh trong khóa học đều giả định bạn đã kích hoạt môi trường ảo `.venv`** (thấy tiền tố
> `(.venv)` ở đầu dòng nhắc). Mỗi lần mở terminal mới, nhớ chạy lại bước (2) để kích hoạt.

---

## 5. Bài tập (Exercises)

1. **Đọc bản đồ.** Không nhìn lại bảng ở mục 2, hãy tự liệt kê 9 bước của pipeline theo thứ tự, và viết
   một câu cho mỗi bước trả lời "vì sao cần bước này?".
2. **Phân biệt hai chế độ.** Với mỗi tình huống sau, đó là *huấn luyện (training)* hay *suy luận
   (inference)*? (a) điều chỉnh tham số để giảm loss; (b) đưa một ảnh mới vào và lấy nhãn dự đoán; (c)
   tính loss trên tập train để biết hướng chỉnh tham số.
3. **Vì sao phải chia dữ liệu?** Giải thích bằng lời của bạn (3–4 câu) tại sao chấm điểm mô hình trên
   **chính dữ liệu nó đã được huấn luyện** lại là một ý tưởng tồi. Dùng phép ẩn dụ "phát trước đề thi".
4. **Sửa `hello.py`.** Thêm hai dòng in ra: tổng các phần tử của `x` (gợi ý: `x.sum()`) và giá trị
   trung bình của `x` (gợi ý: `x.mean()`). Chạy lại và xem kết quả.
5. **(Suy nghĩ)** Phép ẩn dụ "ném phi tiêu bịt mắt" tương ứng với những bước nào trong bản đồ? Cụ thể:
   "đo trượt bao xa" là bước nào, "chỉnh tay" là bước nào?

<details>
<summary>Gợi ý lời giải (solution hints)</summary>

1. Theo đúng bảng ở mục 2: collect → split → preprocess → build model → loss → training loop →
   evaluate → save → inference.
2. (a) training; (b) inference; (c) training (việc tính loss để lấy hướng cập nhật là một phần của
   vòng lặp huấn luyện).
3. Vì mô hình có thể **học thuộc** dữ liệu đó (overfitting) và đạt điểm cao giả tạo, giống học sinh
   được phát trước đề. Điểm số chỉ đáng tin khi đo trên dữ liệu **chưa từng thấy** (test set).
4. `print("sum  =", x.sum())` và `print("mean =", x.mean())`. Kết quả: `sum = tensor(10.)`,
   `mean = tensor(2.)`.
5. "Đo trượt bao xa" ↔ tính **loss** (bước 5/6); "chỉnh tay theo hướng" ↔ tính **gradient** rồi
   **cập nhật tham số** trong vòng lặp huấn luyện (bước 6).

</details>

---

## 6. Tiếp theo (What's next)

Bản đồ đã có trong đầu và môi trường đã sẵn sàng. Ở **Chương 2 — Python & NumPy vừa đủ**, ta sẽ trang bị
đúng những công cụ Python và NumPy (mảng, kích thước/shape, vector hóa, broadcasting) mà từ Chương 3 trở
đi ta sẽ dùng liên tục để biến toán thành code.
