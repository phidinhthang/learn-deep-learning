# ch09_logreg_torch.py
# Dựng lại hồi quy logistic (Chương 5) bằng PyTorch + autograd.
# Cùng dữ liệu, cùng kết quả — nhưng KHÔNG phải tự viết gradient (p - y)*x nữa:
# autograd tự suy ra nó từ công thức loss.
# Cách chạy:  python ch09_logreg_torch.py

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# Dữ liệu giống Chương 5: x = giờ ôn lệch mốc, y = rớt(0)/đậu(1) — đối xứng nên b ở lại ~0
x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device)
y = torch.tensor([0., 0., 0., 1., 1., 1.], device=device)

w = torch.zeros(1, requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)
lr = 0.5

for step in range(9):
    p = torch.sigmoid(w * x + b)                                       # xác suất đậu
    loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()     # BCE (cross-entropy nhị phân)
    if step % 2 == 0:
        print(f"step {step} | loss = {loss.item():.4f} | w = {w.item():.4f}")
    loss.backward()                                                    # autograd tính dloss/dw, dloss/db
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

pred = (torch.sigmoid(w * x + b) > 0.5).float()                        # ngưỡng 0.5 -> nhãn
print("Dự đoán   =", pred.tolist())
print("Nhãn đúng =", y.tolist())
