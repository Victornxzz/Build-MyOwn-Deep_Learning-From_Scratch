from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from src.tensor import Tensor
from src.nn import Dense

def xavier_uniform(
        n_in: int,
        n_out: int,
        *,
        gain: float = 1.0,
        seed: Optional[int] = None
) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    a = gain * np.sqrt(6.0/ (n_in + n_out)).astype(np.float64)
    return rng.uniform(-a, a, size=(n_in, n_out)).astype(np.float64)

def xavier_normal(
        n_in: int,
        n_out: int, 
        *, 
        gain: float = 1.0, 
        seed: Optional[int] = None
) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    std = gain * np.sqrt(2.0/ (n_in + n_out))
    return rng.normal(0.0, std, size=(n_in, n_out)).astype(np.float64)

def he_normal(
        n_in: int,
        n_out: int,
        *,
        seed: Optional[int] = None
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0.0, std, size=(n_in, n_out)).astype(np.float64)

def he_uniform(
        n_in: int,
        n_out: int,
        *,
        seed: Optional[int] = None
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    a = np.sqrt(6.0 / n_in)
    return rng.uniform(-a, a, size=(n_in, n_out)).astype(np.float64)

def init_dense(
        layer: Dense,
        W: np.ndarray,
        b: Optional[np.ndarray] = None
) -> np.ndarray:

    if layer.W.shape != W.shape:
        raise ValueError(f"Incorrect shape for 'W'. Expected {layer.W.shape}, received {W.shape}")

    layer.W.data = W.copy()
    layer.W.grad = np.zeros_like(layer.W.data)

    if b is not None:
        if layer.b is None:
            raise ValueError("The layer does not have a bias, but 'b' was provided.")
        if layer.b.shape != b.shape:
            raise ValueError(f"Incorrect shape for b. Expected {layer.b.shape}, received {b.shape}")

        layer.b.data = b.copy()
        layer.b.grad = np.zeros_like(layer.b.data)

def forward_activation_stats(
        sizes: Sequence[int],
        init_fn: Callable[[int, int], np.ndarray],
        activation: str,
        *,
        n_samples: int = 512,
        seed: Optional[int] = None
) -> List[Dict[str, float]]:

    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(sizes) -1):
        n_in, n_out = sizes[i], sizes[i + 1]
        layer = Dense(n_in, n_out, bias=True)

        W = init_fn(n_in, n_out)
        b = np.zeros((n_out), dtype=np.float64)
        init_dense(layer, W, b)

        layers.append(layer)

    X = Tensor(rng.normal(0.0, 1.0, size=(n_samples, sizes[0])).astype(np.float64))

    stats = []
    out = X
    for layer in layers:
        out = layer(out)
        if activation == "tanh":
            out = out.tanh()
        elif activation == "relu":
            out = out.relu()

        data = out.data
        stats.append({
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "saturated": float(np.mean(np.abs(data) > 0.98)),
            "dead": float(np.mean(data == 0.0))
        })

    return stats