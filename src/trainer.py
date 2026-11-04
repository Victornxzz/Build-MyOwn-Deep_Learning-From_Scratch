from typing import Dict, List, Optional, Union
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