# ch09_autograd.py
# Làm quen TENSOR và AUTOGRAD của PyTorch, rồi dựng lại hồi quy tuyến tính (Chương 4)
# — lần này để autograd TỰ tính phần backward thay vì ta viết tay như Chương 7.
# Cách chạy:  python ch09_autograd.py

import torch

# ---- (A) Tensor: họ hàng của mảng NumPy, nhưng chạy được GPU và biết tự tính gradient ----
device = "cuda" if torch.cuda.is_available() else "cpu"   # Chương 10 sẽ gói lại thành get_device()
print("device =", device)

a = torch.tensor([[1., 2.], [3., 4.]], device=device)
print("a.shape =", a.shape, "| a.dtype =", a.dtype)
print("a + 10  =\n", a + 10)              # phép toán theo từng phần tử (giống NumPy)
print("a @ a   =\n", a @ a)               # nhân ma trận
print("sang numpy:", a.cpu().numpy().ravel())   # đổi sang NumPy (phải đưa về CPU trước)

# ---- (B) Autograd: tự động tính đạo hàm ----
# Kiểm chứng tay: f(x) = x^3 thì f'(x) = 3x^2; tại x=2 -> 12.
x = torch.tensor(2.0, requires_grad=True)   # requires_grad=True: "hãy theo dõi x để tính gradient"
f = x ** 3
f.backward()                                # tính df/dx, lưu vào x.grad
print("df/dx tại x=2 =", x.grad.item(), "(đúng 3*x^2 = 12)")

# ---- (C) Dựng lại hồi quy tuyến tính Chương 4 — autograd lo phần backward ----
torch.manual_seed(42)
xs = torch.tensor([-2., -1., 0., 1., 2.], device=device)
ys = torch.tensor([12., 16., 20., 24., 28.], device=device)

w = torch.zeros(1, requires_grad=True, device=device)   # tham số cần học
b = torch.zeros(1, requires_grad=True, device=device)
lr = 0.1

for step in range(51):
    y_pred = w * xs + b                       # forward
    loss = ((y_pred - ys) ** 2).mean()        # MSE
    if step % 10 == 0:
        print(f"step {step:3d} | loss = {loss.item():.6f} | w = {w.item():.4f}, b = {b.item():.4f}")
    loss.backward()                           # AUTOGRAD: tự tính dloss/dw, dloss/db
    with torch.no_grad():                     # cập nhật tham số — KHÔNG theo dõi gradient ở bước này
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()                        # xóa gradient cũ (autograd CỘNG DỒN nếu không xóa)
        b.grad.zero_()

print(f"Học xong: w = {w.item():.4f}, b = {b.item():.4f}  (quan hệ thật: w=4, b=20)")
print(f"Dự đoán cho x=1.5: {(w * 1.5 + b).item():.4f}  (đúng phải là 26)")
