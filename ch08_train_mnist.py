# ch08_train_mnist.py
# Huấn luyện MLP viết tay (NumPy) để phân loại chữ số MNIST — quy trình ĐẦY ĐỦ:
# nạp dữ liệu -> chia train/val -> lô nhỏ (mini-batch) -> nhiều epoch -> đánh giá trên test.
# Cách chạy:  python ch08_train_mnist.py
# Lần đầu chạy sẽ TỰ TẢI MNIST (~12 MB) về thư mục ./data (cần mạng).

import numpy as np
from ddl.data import load_mnist, train_val_test_split, iterate_minibatches
from ddl.numpy_nn import Linear, ReLU, Sequential, SoftmaxCrossEntropy, sgd_step
from ddl.metrics import predict_labels, accuracy

np.random.seed(42)   # tái lập kết quả

# (1) NẠP dữ liệu: ảnh duỗi thành vector 784, pixel về [0, 1]
(X_train_full, y_train_full), (X_test, y_test) = load_mnist(flatten=True, normalize=True)

# (2) CHIA train -> train + val (val để theo dõi mô hình trên dữ liệu nó chưa học)
(X_train, y_train), (X_val, y_val), _ = train_val_test_split(
    X_train_full, y_train_full, val_fraction=0.1, seed=42)
print(f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# (3) XÂY mô hình: 784 -> 128 (ReLU) -> 10 logits (một logit cho mỗi chữ số 0..9)
net = Sequential([
    Linear(784, 128, seed=42),
    ReLU(),
    Linear(128, 10, seed=43),
])
loss_fn = SoftmaxCrossEntropy()
lr, n_epochs, batch_size = 0.1, 5, 64

# (4) VÒNG LẶP HUẤN LUYỆN: mỗi epoch duyệt HẾT tập train theo từng lô nhỏ
for epoch in range(1, n_epochs + 1):
    running_loss, n_batches = 0.0, 0
    for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size,
                                                shuffle=True, seed=epoch):
        logits = net.forward(X_batch)              # forward trên LÔ
        loss = loss_fn.forward(logits, y_batch)    # mất mát trên lô
        net.backward(loss_fn.backward())           # lan truyền ngược -> điền dW, db
        sgd_step(net.parameters(), lr)             # cập nhật tham số
        running_loss += loss
        n_batches += 1
    # Sau mỗi epoch: đo độ chính xác trên tập VAL (dữ liệu mô hình không học)
    val_acc = accuracy(predict_labels(net.forward(X_val)), y_val)
    print(f"epoch {epoch} | train_loss = {running_loss / n_batches:.4f} | val_acc = {val_acc:.4f}")

# (5) ĐÁNH GIÁ CUỐI trên tập TEST (dữ liệu mô hình CHƯA hề thấy lần nào)
test_acc = accuracy(predict_labels(net.forward(X_test)), y_test)
print(f"Độ chính xác trên TEST = {test_acc:.4f}")
