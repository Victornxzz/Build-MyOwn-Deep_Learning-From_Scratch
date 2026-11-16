from typing import Dict, List, Optional, Union, Iterator, Mapping, Sequence, Tuple
import numpy as np

from src.tensor import Tensor
from src.nn import MLP
from src.loss import mse_loss
from src.optim import SGD, Optimizer


def accuracy(pred: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> float:
    p_data = np.asarray (pred.data if hasattr(pred, "data") else pred)
    y_data = np.asarray (y.data if hasattr(y, "data") else y)

    if p_data.size != y_data.size:
        raise ValueError(f"Imcompatible sizes for accuracy: pred={p_data.size}, y={y_data.size}")

    p_flat = p_data.flatten()
    y_flat = y_data.flatten()

    correct = (np.sign(p_flat) == np.sign(y_flat))
    return float(np.mean(correct))

def train(
        model: MLP,
        X: Tensor,
        y: Tensor,
        *,
        lr: float = 0.1,
        epochs: int = 200,
        optimizer: Optional[Optimizer] = None,
) -> Dict[str, List[float]]:

    if not isinstance(X, Tensor) or not isinstance(y, Tensor):
        raise TypeError(f"X and Y must be instaces of Tensor")

    if len(X.shape) != 2:
        raise ValueError(f"X must be 2-D (N, n_in), X.shape : {X.shape}")

    if len(y.shape) not in (1, 2) or (len(y.shape) == 2 and y.shape[1] != 1):
        raise ValueError(f"Y must be (N,) or (N, 1), y.shape : {y.shape}")

    N = X.shape[0]
    if y.shape[0] != N:
        raise ValueError(f"Lines divergence: X has {N}, y has {y.shape[0]}")

    y_col_data = y.data.reshape(N, 1)
    y_target = Tensor(y_col_data)

    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = mse_loss(preds, y_target)

        history["loss"].append(float(loss.data))
        history["accuracy"].append(accuracy(preds, y_target))

        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    return history

def plot_history(history: Dict[str, List[float]], path: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_axis = range(1, len(history["loss"]) + 1)

    ax_loss.plot(epochs_axis, history["loss"])
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")

    ax_acc.plot(epochs_axis, history["loss"])
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_title("Training accuracy")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    return fig

def iterate_minibatches(
    X: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Gerador que particiona (X, y) em fatias (X_b, y_b) de tamanho batch_size."""
    X_arr = np.asarray(X.data if hasattr(X, "data") else X)
    y_arr = np.asarray(y.data if hasattr(y, "data") else y)

    assert X_arr.ndim >= 1 and y_arr.ndim >= 1, "X e y devem ter pelo menos 1 dimensão."

    N = X_arr.shape[0]
    assert y_arr.shape[0] == N, f"X e y devem ter o mesmo número de linhas: {N} != {y_arr.shape[0]}"
    assert isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool), "batch_size deve ser inteiro."
    assert 1 <= batch_size <= N, f"batch_size deve estar entre 1 e N={N}, recebido: {batch_size}"

    indices = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    for start in range(0, N, batch_size):
        end = start + batch_size
        if end > N and drop_last:
            break
        batch_idx = indices[start:end]
        yield X_arr[batch_idx], y_arr[batch_idx]


def train_minibatch(
    model: MLP,
    X: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
    *,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    optimizer: Optional[Optimizer] = None,
    drop_last: bool = False,
) -> Dict[str, object]:
    """Treina o modelo iterando sobre mini-batches a cada época."""
    X_arr = np.asarray(X.data if hasattr(X, "data") else X)
    y_arr = np.asarray(y.data if hasattr(y, "data") else y)

    if X_arr.ndim != 2:
        raise ValueError(f"X precisa ser 2-D (N, n_in), recebido: {X_arr.shape}")
    if y_arr.ndim not in (1, 2) or (y_arr.ndim == 2 and y_arr.shape[1] != 1):
        raise ValueError(f"y precisa ser (N,) ou (N, 1), recebido: {y_arr.shape}")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(f"Divergência de linhas: X tem {X_arr.shape[0]}, y tem {y_arr.shape[0]}")

    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    batch_loss_history: List[float] = []
    epoch_loss_history: List[float] = []
    steps = 0

    for _ in range(epochs):
        epoch_seed = int(rng.integers(0, 2**31 - 1)) if shuffle else None
        weighted_loss_sum = 0.0
        total_samples = 0

        for X_b, y_b in iterate_minibatches(
            X_arr, y_arr, batch_size, shuffle=shuffle, seed=epoch_seed, drop_last=drop_last
        ):
            B = X_b.shape[0]
            X_t = Tensor(X_b)
            y_t = Tensor(y_b.reshape(B, 1))

            optimizer.zero_grad()
            pred = model(X_t)
            loss = mse_loss(pred, y_t)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.data)
            batch_loss_history.append(loss_val)
            weighted_loss_sum += loss_val * B
            total_samples += B
            steps += 1

        if total_samples > 0:
            epoch_loss_history.append(weighted_loss_sum / total_samples)

    optimizer.zero_grad()
    return {
        "batch_loss": batch_loss_history,
        "epoch_loss": epoch_loss_history,
        "steps": steps,
    }


def gradient_noise(
    model: MLP,
    X: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
    batch_size: int,
    *,
    n_batches: int,
    seed: Optional[int] = None,
) -> float:
    """Estima o ruído do gradiente estocástico para um dado batch_size sem mover os pesos."""
    if n_batches <= 0:
        raise ValueError(f"n_batches deve ser > 0, recebido: {n_batches}")

    X_arr = np.asarray(X.data if hasattr(X, "data") else X)
    y_arr = np.asarray(y.data if hasattr(y, "data") else y)
    N = X_arr.shape[0]

    if y_arr.shape[0] != N:
        raise ValueError(f"Divergência de linhas: X tem {N}, y tem {y_arr.shape[0]}")
    if batch_size <= 0 or batch_size > N:
        raise ValueError(f"batch_size deve estar entre 1 e N={N}, recebido: {batch_size}")

    rng = np.random.default_rng(seed)
    params = list(model.parameters())
    grad_vectors: List[np.ndarray] = []

    for _ in range(n_batches):
        # Zera gradientes manualmente sem precisar de um otimizador
        for p in params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data, dtype=np.float64)

        idx = rng.choice(N, size=batch_size, replace=False)
        X_t = Tensor(X_arr[idx])
        y_t = Tensor(y_arr[idx].reshape(batch_size, 1))

        pred = model(X_t)
        loss = mse_loss(pred, y_t)
        loss.backward()

        # Achata e concatena todos os gradientes dos parâmetros em um único vetor 1-D
        flat_g = np.concatenate([
            (p.grad if p.grad is not None else np.zeros_like(p.data)).ravel()
            for p in params
        ])
        grad_vectors.append(flat_g)

    # Limpa os gradientes ao sair para manter o modelo higiênico
    for p in params:
        if p.grad is not None:
            p.grad = np.zeros_like(p.data, dtype=np.float64)

    # Empilha (n_batches, n_params) e tira a média da variância por coordenada
    stacked = np.stack(grad_vectors, axis=0)
    per_coord_var = np.var(stacked, axis=0)
    return float(np.mean(per_coord_var))


def epochs_to_threshold(history: Sequence[float], threshold: float) -> int:
    """Retorna o índice 1-based da primeira época com loss <= threshold, ou -1 se nunca atingir."""
    for idx, val in enumerate(history, start=1):
        if val <= threshold:
            return idx
    return -1


def plot_batch_comparison(
    histories: Mapping[str, Sequence[float]],
    path: Optional[str] = None,
):
    """Plota múltiplas curvas de epoch_loss rotuladas no mesmo gráfico."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, losses in histories.items():
        epochs_axis = range(1, len(losses) + 1)
        ax.plot(epochs_axis, losses, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Mini-batch loss comparison")
    ax.legend()
    fig.tight_layout()

    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    return fig