# Course Contract — Deep Learning từ số 0 (from scratch)

> This file is the **single source of truth** for notation, file names, code signatures, and writing
> conventions. Every chapter MUST obey it. It exists so that 14 independently-written chapters stay
> consistent (same symbols, same function names, same shapes) even though no code is ever executed
> during authoring. When in doubt while writing a chapter, this file wins.

---

## 1. Course identity

- **Topic:** Deep learning from scratch — from "what is a gradient" up to a working Convolutional
  Neural Network (CNN) on MNIST.
- **One-line goal:** By the end, the learner can explain the full training pipeline, hand-derive and
  hand-code backpropagation for a small network in NumPy, and build/train/evaluate/run an MLP and a
  CNN in PyTorch on MNIST.
- **Scope — in:**
  - The end-to-end pipeline: collect data → split → preprocess → build model → loss → train →
    evaluate → save → inference.
  - Math foundations actually used: vectors/matrices, derivatives, gradients, the chain rule.
  - Models: linear regression, logistic regression, MLP (from scratch in NumPy), then MLP & a small
    CNN in PyTorch.
  - Core training ideas: gradient descent, mini-batches, epochs, overfitting, train/val/test split,
    learning rate, regularization basics (dropout), normalization/standardization.
- **Scope — out (explicitly):** RNNs/LSTMs, attention/Transformers, NLP, generative models, GPUs at
  scale/distributed training, deployment/serving, advanced optimizers theory, autograd internals
  beyond an intuitive "what it automates" explanation. (These are named as "next steps" in Ch 14.)

## 2. Audience & depth

- **Assumed background:** New to deep learning and PyTorch. **Basic** Python (can read simple code,
  not fluent) and **rusty** math (calculus/linear algebra seen long ago). So: explain Python idioms
  when first used; re-teach each piece of math intuitively before using it.
- **Theory rigor:** Rigorous but intuition-first. Every formula is *motivated* before it appears, and
  derivations are shown step-by-step (especially backprop in Ch 7), never hand-waved. Always pair the
  math with a tiny worked numeric example.
- **Math level:** Standard notation (below), introduced gently. No measure theory, no proofs for
  their own sake. Each new symbol is defined in words the first time it appears in a chapter.

## 3. Tools & substrate

- **Language / framework:** Python 3.10+. NumPy for the "from scratch" parts; PyTorch (+ torchvision)
  from Ch 9 on. Matplotlib for plots.
- **From-scratch boundary:**
  - Ch 4–8 (NumPy era): **build by hand** — forward pass, loss, gradients, and backpropagation. The
    only library allowed is NumPy (plus `torchvision.datasets.MNIST` used *only as a download helper*
    to fetch raw pixel arrays in Ch 8, clearly flagged). No autograd, no `nn`, no optimizers.
  - Ch 9–14 (PyTorch era): PyTorch tensors, `torch.autograd`, `torch.nn`, `torch.optim`,
    `torch.utils.data`, `torchvision` are all allowed. The from-scratch **conv forward** pass in
    Ch 12 is hand-coded in NumPy for intuition, but conv's **backward** is left to autograd.
  - Never allowed (would defeat the point): `nn.Transformer`, pretrained models, high-level training
    wrappers (e.g. PyTorch Lightning, `skorch`), `sklearn` models. We implement train/val/test split
    ourselves.
- **Concrete substrate:**
  - Synthetic/toy data (generated with NumPy) for building intuition in Ch 4–6.
  - **MNIST** handwritten digits (28×28 grayscale, 10 classes) as the one "real" dataset, used flat
    (vectors of length 784) for the MLP and as images (1×28×28) for the CNN.
- **Environment / hardware:** Device-agnostic. Code runs on a laptop **CPU** by default and uses a
  CUDA **GPU** automatically if present, via `get_device()` (see §5). Models/datasets kept small
  enough to train in seconds–minutes on CPU.

## 4. Notation & conventions

### 4.1 Math symbols (fixed meanings, used throughout)

| Symbol | Meaning |
|--------|---------|
| $x$, $\mathbf{x}$, $X$ | scalar / vector (bold lowercase) / matrix (uppercase) input |
| $\mathbf{x}^{(i)}$ | the $i$-th training example |
| $y$, $\hat{y}$ | true target / model prediction (read "y-hat") |
| $N$ | number of examples in the current batch/dataset |
| $D$ | number of input features; $K$ = number of classes (here $K=10$) |
| $W$, $\mathbf{b}$ | weight matrix, bias vector of a layer |
| $z$ | pre-activation (before the nonlinearity), $z = W^\top x + b$ |
| $a = \sigma(z)$ | activation (after the nonlinearity); $\sigma$ = sigmoid, $\text{ReLU}$, softmax as named |
| $\mathcal{L}$, $J$ | per-example loss, and $J$ = mean loss over the batch (the cost) |
| $\eta$ | learning rate ("eta") |
| $\theta$ | all trainable parameters collectively |
| $\nabla_\theta J$ | gradient of the cost w.r.t. parameters |
| superscript $[l]$ | quantities of layer $l$ (e.g. $W^{[1]}$, $a^{[2]}$) |

- Math in Markdown uses `$...$` (inline) and `$$...$$` (display).
- For each new symbol, state its meaning **and its shape** the first time it appears in a chapter.

### 4.2 Array / tensor shapes (batch-first, always)

- Tabular / flattened input: shape `(N, D)`. MNIST flattened: `(N, 784)`.
- Labels: integer class indices, shape `(N,)` (NOT one-hot, unless explicitly building one-hot).
- Images in PyTorch: `(N, C, H, W)` = `(N, 1, 28, 28)` for MNIST.
- Layer weight `W` in our **NumPy** layers has shape `(in_features, out_features)` so that the forward
  pass is `z = x @ W + b` with `x` of shape `(N, in_features)`. (Note: PyTorch's `nn.Linear` stores
  its weight transposed; we point this out in Ch 10 so the learner isn't confused.)

### 4.3 Language & code conventions

- **Prose/theory is in Vietnamese.** The first time an important technical term appears in a chapter,
  write it in Vietnamese with the **English term in parentheses**, e.g. "hàm mất mát (loss function)",
  "lan truyền ngược (backpropagation)". After that, either may be used.
- **Code and identifiers are in English, but explanatory comments are in Vietnamese** so the learner
  can read them — with the English term in parentheses on first use. Comments are heavy and tie each
  line back to the math, e.g. `# đạo hàm của J theo W (gradient w.r.t. W): dW = x^T @ grad`.
- **Identifier style:** `snake_case` for functions/variables, `PascalCase` for classes. Conventional
  names kept: feature matrix `X` (capital), labels `y` (lowercase), `y_pred` / `y_hat`, `lr`,
  `n_epochs`, `batch_size`, `model`, `loss_fn`, `optimizer`, `loader`, `device`.
- **Reproducibility:** fixed seed `42` everywhere (`np.random.seed(42)`, `torch.manual_seed(42)`).
- **Expected outputs:** Because code is never executed during authoring, every "Run it" block shows a
  *plausible* expected output and is annotated that exact numbers may differ slightly across machines
  (note in Vietnamese: "kết quả của bạn có thể hơi khác (your numbers may vary a little)").

## 5. The shared code project (the anti-drift contract)

All reusable code lives in a package named **`ddl/`** ("Deep-dive Deep Learning"), created in the
course root. Chapters contribute files to it incrementally; later chapters `import` from it. Names and
signatures below are **fixed** — a chapter must use them verbatim.

| File | Introduced in | Public surface (must match exactly) |
|------|---------------|-------------------------------------|
| `ddl/__init__.py` | Ch 2 | empty package marker |
| `ddl/preprocessing.py` | Ch 8 | `class Standardizer:` with `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(X)`; stores `self.mean_`, `self.std_`. (Originally slated for Ch 4, but moved here so the first model stays minimal; standardization is introduced where MNIST/multiple features make it necessary.) |
| `ddl/functions.py` | Ch 5 | `sigmoid(z)`, `softmax(z)`, `relu(z)`; losses `mse_loss(y_hat, y)`, `bce_loss(p, y)` (all NumPy) |
| `ddl/numpy_nn.py` | Ch 6–7 | `class Linear(in_features, out_features)`, `class ReLU()`, `class Sequential(layers)`, `class SoftmaxCrossEntropy()`; each layer has `.forward(x)`, `.backward(grad)`, `.parameters()`; module-level `sgd_step(params, lr)` |
| `ddl/data.py` | Ch 8 | `load_mnist(flatten=True, normalize=True)` → `(X_train, y_train), (X_test, y_test)` (NumPy); `train_val_test_split(X, y, val_fraction=0.1, test_fraction=0.0, seed=42)`; `iterate_minibatches(X, y, batch_size, shuffle=True, seed=None)` generator |
| `ddl/metrics.py` | Ch 8 | `predict_labels(logits)` (argmax over last axis); `accuracy(y_pred_labels, y_true)` |
| `ddl/config.py` | Ch 10 | `get_device()` → `"cuda"`/`"cpu"`; `@dataclass Config` (fields in §5.1); `default_config()` |
| `ddl/torch_data.py` | Ch 10 | `get_mnist_loaders(config)` → `(train_loader, val_loader, test_loader)` |
| `ddl/models.py` | Ch 10, 13 | `class MLP(nn.Module)` (Ch 10, `__init__(in_features=784, hidden_dim=128, num_classes=10)`); `class SmallCNN(nn.Module)` (Ch 13, `__init__(num_classes=10)`); both define `forward(x)` |
| `ddl/engine.py` | Ch 11 | `train_one_epoch(model, loader, loss_fn, optimizer, device)` → `avg_loss`; `evaluate(model, loader, loss_fn, device)` → `(avg_loss, acc)`; `fit(config, model, train_loader, val_loader, loss_fn, optimizer)` → `history` dict |
| `ddl/checkpoint.py` | Ch 11 | `save_checkpoint(model, path)`; `load_checkpoint(model, path, device)` → `model` |

### 5.1 Central config object (one object passed everywhere)

```python
# ddl/config.py
from dataclasses import dataclass, field
import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    seed: int = 42
    device: str = field(default_factory=get_device)
    data_dir: str = "./data"
    batch_size: int = 64
    lr: float = 0.1            # SGD default; chapters using Adam override to 1e-3
    n_epochs: int = 5
    hidden_dim: int = 128
    num_classes: int = 10
    val_fraction: float = 0.1

def default_config():
    return Config()
```

- Chapters create a config with `cfg = default_config()` and override fields explicitly
  (e.g. `cfg = Config(lr=1e-3, n_epochs=10)`), never hard-coding hyperparameters in two places.
- The NumPy chapters (4–8) predate `Config`; they use plain local variables and the contract notes
  this is intentional (we introduce a config object once we reach the "engineering" PyTorch stage).

## 6. Per-chapter skeleton (exact section headings)

Every chapter file is `chapters/NN-slug.md` and uses these top-level (`##`) sections in order:

1. `## 0. Chúng ta đang ở đâu (Where we are)` — recap of prior chapters + a one-line "map" callback
   showing which pipeline step this chapter is about, and what it adds.
2. `## 1. Trực giác & ý tưởng (Intuition & idea)` — the why, motivated, before any formula.
3. `## 2. Toán học (The math)` — rigorous, step-by-step, with a tiny numeric example.
4. `## 3. Cài đặt (Implementation)` — runnable, heavily-commented code obeying §4–§5.
5. `## 4. Chạy thử (Run it)` — copy-paste commands + expected output (annotated as approximate).
6. `## 5. Bài tập (Exercises)` — 3–5 exercises, each with a short solution hint at the end.
7. `## 6. Tiếp theo (What's next)` — one or two sentences pointing to the next chapter.

The very top of each file: `# Chương N — <Vietnamese title> (<English title>)`.

## 7. Hard rules (re-read before writing any chapter)

- Output is **Markdown only**. Never install, download, run, or validate by executing. Any command
  the learner runs goes **into the chapter** as a copy-paste block with annotated expected output.
- Build "from-scratch" parts by hand within the §3 boundary; use only allowed primitives per era.
- Code must be **internally self-consistent**: names, signatures, and shapes match §4–§5 and match
  what earlier chapters established. Self-review shapes before emitting.
- Keep the textbook voice: explain **why**, then **how**. Theory **and code comments** in Vietnamese
  (+ English term in parentheses on first use); code/identifiers in English.
- Each chapter must explicitly re-anchor to the **pipeline map** from Ch 1 (§0 section).
- Prefer small, fast examples (CPU-friendly). Keep MNIST runs to a few epochs.

## 8. Chapter list (confirmed with the user)

**Part 0 — Orientation & setup**
1. Bức tranh tổng thể (The big picture) — what DL is, the full pipeline & why each step exists, setup.
2. Python & NumPy vừa đủ (Python & NumPy just enough) — Python basics + arrays, shapes, broadcasting.

**Part 1 — The first learning algorithm, by hand in NumPy**
3. Toán cần thiết (The math we actually need) — vectors, matrices, derivatives, gradients, chain rule.
4. Hồi quy tuyến tính từ số 0 (Linear regression from scratch) — model + MSE + gradient descent + loop.
5. Phân loại nhị phân từ số 0 (Logistic regression from scratch) — sigmoid, cross-entropy, boundary.

**Part 2 — Neural networks from scratch in NumPy**
6. Từ nơ-ron đến mạng (From a neuron to a network) — MLP, activations, why nonlinearity, forward pass.
7. Lan truyền ngược bằng tay (Backpropagation by hand) — chain rule layer-by-layer; forward + backward.
8. Huấn luyện MLP & quy trình thực tế (Training the MLP + the real pipeline) — split, batches, epochs, MNIST.

**Part 3 — Re-doing it the PyTorch way**
9. Tensor và autograd (Tensors & autograd) — PyTorch tensors, autograd, rebuild Ch 4–5, device-agnostic.
10. Mô hình kiểu PyTorch (Models the PyTorch way) — `nn.Module`, optimizers, `Dataset`/`DataLoader`.
11. Vòng lặp huấn luyện, đánh giá & suy luận (Training loop, evaluation & inference) — full loop, save/load.

**Part 4 — Convolutional neural networks**
12. Tại sao cần CNN (Why CNNs) — limits of MLP on images, convolution (math + from-scratch forward).
13. Các khối của CNN (CNN building blocks) — `nn.Conv2d`, pooling, padding/stride, output-size math.
14. Huấn luyện CNN trên MNIST, trọn vẹn (Train a CNN on MNIST, end-to-end) — full pipeline, dropout, next steps.
