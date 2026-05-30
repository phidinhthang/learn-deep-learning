"""ddl/data.py — nạp dữ liệu MNIST và chia lô (mini-batch), viết bằng NumPy.

MNIST = 70.000 ảnh chữ số viết tay 28x28 mức xám, 10 lớp (0..9): 60.000 ảnh huấn
luyện + 10.000 ảnh kiểm tra. Ta CHỈ mượn torchvision để TẢI ảnh thô về máy
(download helper), rồi tự xử lý mọi thứ bằng NumPy — đúng tinh thần "from scratch".
"""

import numpy as np


def load_mnist(flatten=True, normalize=True):
    """Trả về (X_train, y_train), (X_test, y_test) dạng mảng NumPy.

    flatten=True   -> mỗi ảnh duỗi (flatten) thành vector 784 = 28*28; nếu không, giữ (N, 28, 28).
    normalize=True -> chia 255 để pixel về đoạn [0, 1] (giúp huấn luyện ổn định hơn).
    """
    from torchvision import datasets   # CHỈ dùng để TẢI dữ liệu thô về (không dùng model/optim)

    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)

    X_train = train.data.numpy().astype(np.float32)    # (60000, 28, 28), pixel 0..255
    y_train = train.targets.numpy().astype(np.int64)   # (60000,), nhãn 0..9
    X_test = test.data.numpy().astype(np.float32)      # (10000, 28, 28)
    y_test = test.targets.numpy().astype(np.int64)     # (10000,)

    if normalize:
        X_train = X_train / 255.0      # đưa pixel về [0, 1]
        X_test = X_test / 255.0
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)   # (60000, 784)
        X_test = X_test.reshape(X_test.shape[0], -1)      # (10000, 784)
    return (X_train, y_train), (X_test, y_test)


def train_val_test_split(X, y, val_fraction=0.1, test_fraction=0.0, seed=42):
    """Xáo trộn rồi cắt dữ liệu thành train / val / test.

    Trả về ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    Vì sao tách val? Để theo dõi mô hình trên dữ liệu nó CHƯA học, phát hiện học vẹt
    (overfitting). Nếu test_fraction=0 thì phần test rỗng (MNIST đã có tập test riêng).
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)          # một thứ tự ngẫu nhiên CỐ ĐỊNH theo seed
    n_val = int(N * val_fraction)
    n_test = int(N * test_fraction)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return ((X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx]))


def iterate_minibatches(X, y, batch_size, shuffle=True, seed=None):
    """Sinh lần lượt từng lô nhỏ (X_batch, y_batch) cho MỘT lượt qua dữ liệu (epoch).

    Dùng làm generator:  for X_batch, y_batch in iterate_minibatches(...):
    shuffle=True -> xáo lại thứ tự mỗi epoch để các lô mỗi lần một khác.
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        batch = idx[start:start + batch_size]   # chỉ số của các mẫu trong lô này
        yield X[batch], y[batch]
