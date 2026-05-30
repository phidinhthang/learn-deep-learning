# Deep Learning từ số 0 (Deep Learning from scratch)

Một khóa học (course) đưa bạn từ con số 0 — chưa biết gì về deep learning hay PyTorch — đến chỗ tự
tay xây, huấn luyện (train) và chạy một mạng nơ-ron tích chập (Convolutional Neural Network — CNN)
nhận diện chữ số viết tay.

Triết lý của khóa học:

- **Hiểu trước, dùng sau.** Mỗi khái niệm đều bắt đầu bằng *trực giác (intuition)* và lý do "tại sao
  ta cần nó", rồi mới đến công thức toán, rồi mới đến code.
- **Tự làm từ số 0.** Ở phần đầu, ta tự viết tay (in NumPy) cả lượt tính xuôi (forward pass) lẫn lan
  truyền ngược (backpropagation) — để bạn thấy rõ "bên trong" một mạng nơ-ron. Sau đó mới chuyển sang
  PyTorch để thấy nó tự động hóa đúng những thứ ta vừa làm tay.
- **Luôn nhìn thấy bức tranh tổng thể.** Toàn bộ quy trình huấn luyện (training pipeline) được giới
  thiệu ngay ở Chương 1, và mỗi chương sau đều nhắc lại "ta đang ở bước nào trên bản đồ".

> Lý thuyết viết bằng tiếng Việt, kèm thuật ngữ tiếng Anh trong ngoặc (...). Code và tên biến
> (identifiers) viết bằng tiếng Anh, nhưng phần chú thích (comments) giải thích thì viết bằng tiếng
> Việt để người học dễ hiểu.

---

## Cần chuẩn bị gì (Prerequisites)

- Python cơ bản (đọc được code đơn giản). Chương 2 sẽ ôn lại phần Python cần dùng.
- Toán phổ thông; phần giải tích/đại số tuyến tính cần thiết sẽ được ôn lại ở Chương 3.
- Một máy tính chạy được Python 3.10+ (chỉ cần CPU; có GPU thì code tự dùng).

Cài đặt môi trường chi tiết nằm ở **Chương 1**.

## Cấu trúc thư mục (Project layout)

```
learn-deep-learning/
├── README.md              # file này
├── COURSE_CONTRACT.md     # quy ước ký hiệu & code (cho người soạn khóa học)
├── chapters/              # nội dung từng chương (.md)
│   ├── 01-big-picture.md
│   └── ...
└── ddl/                   # package code dùng chung, xây dần qua các chương
```

## Lộ trình (Roadmap) — 14 chương, 4 phần

**Phần 0 — Định hướng & cài đặt**
1. **Bức tranh tổng thể (The big picture)** — deep learning là gì, toàn bộ pipeline & vì sao cần từng
   bước, cài đặt môi trường.
2. **Python & NumPy vừa đủ (Python & NumPy just enough)** — Python cơ bản + mảng (arrays), kích thước
   (shapes), vector hóa (vectorization), broadcasting.

**Phần 1 — Thuật toán học đầu tiên, làm tay bằng NumPy**
3. **Toán cần thiết (The math we actually need)** — vector, ma trận, đạo hàm, gradient, quy tắc chuỗi.
4. **Hồi quy tuyến tính từ số 0 (Linear regression from scratch)** — model + hàm mất mát + gradient
   descent + vòng lặp huấn luyện.
5. **Phân loại nhị phân từ số 0 (Logistic regression from scratch)** — sigmoid, cross-entropy.

**Phần 2 — Mạng nơ-ron từ số 0 bằng NumPy**
6. **Từ nơ-ron đến mạng (From a neuron to a network)** — MLP, hàm kích hoạt, vì sao cần phi tuyến.
7. **Lan truyền ngược bằng tay (Backpropagation by hand)** — quy tắc chuỗi qua từng tầng.
8. **Huấn luyện MLP & quy trình thực tế (Training the MLP + the real pipeline)** — chia dữ liệu, batch,
   epoch, MNIST.

**Phần 3 — Làm lại theo cách PyTorch**
9. **Tensor và autograd (Tensors & autograd)** — tensor, đạo hàm tự động, dựng lại Chương 4–5.
10. **Mô hình kiểu PyTorch (Models the PyTorch way)** — `nn.Module`, optimizer, `DataLoader`.
11. **Vòng lặp huấn luyện, đánh giá & suy luận (Training loop, evaluation & inference)** — lưu/nạp model.

**Phần 4 — Mạng nơ-ron tích chập (CNN)**
12. **Tại sao cần CNN (Why CNNs)** — giới hạn của MLP với ảnh, phép tích chập (convolution).
13. **Các khối của CNN (CNN building blocks)** — `nn.Conv2d`, pooling, padding/stride.
14. **Huấn luyện CNN trên MNIST, trọn vẹn (Train a CNN on MNIST, end-to-end)** — pipeline đầy đủ, các
    bước tiếp theo.

---

## Cách học (How to use this course)

Đọc lần lượt từng chương trong `chapters/`. Mỗi chương có cùng một bố cục:

0. **Chúng ta đang ở đâu (Where we are)** — nhắc lại & nối tiếp chương trước.
1. **Trực giác & ý tưởng (Intuition & idea)** — tại sao cần thứ này.
2. **Toán học (The math)** — công thức kèm ví dụ số nhỏ.
3. **Cài đặt (Implementation)** — code chạy được, chú thích kỹ.
4. **Chạy thử (Run it)** — lệnh sao chép-dán & kết quả mong đợi.
5. **Bài tập (Exercises)** — vài bài luyện tập kèm gợi ý.
6. **Tiếp theo (What's next)** — dẫn sang chương sau.

Bắt đầu từ **[Chương 1 — Bức tranh tổng thể](chapters/01-big-picture.md)**.
