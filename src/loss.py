import numpy as np
from src.tensor import Tensor

def mse_loss(pred, target) -> Tensor:
    pred_t = pred if isinstance(pred, Tensor) else Tensor(pred)
    target_t = target if isinstance (target, Tensor) else Tensor(target)
    diff = pred_t - target_t
    return (diff ** 2).mean()

def mae_loss(pred, target) -> Tensor:
    pred_t = pred if isinstance(pred, Tensor) else Tensor(pred)
    target_t = target if isinstance (target, Tensor) else Tensor(target)
    diff = pred_t - target_t
    abs_diff = diff.relu() + (-diff).relu()
    return abs_diff.mean()

def log_softmax(logits) -> Tensor:
    logits_t = logits if isinstance(logits, Tensor) else Tensor(logits)

    m_val = float(np.max(logits_t.data))
    shift = logits_t - m_val
    lse = shift.exp().sum().log()
    return shift - lse

def softmax(logits) -> Tensor:
    return log_softmax(logits).exp()

def cross_entropy_loss(logits, target) -> Tensor:
    logits_t = logits if isinstance(logits, Tensor) else Tensor(logits)
    n_classes = logits_t.shape[0]

    if isinstance(target, (int, np.integer)):
        y_data = np.zeros(n_classes, dtype=np.float64)
        y_data[target] = 1.0
        y_t = Tensor(y_data)
    else:
        y_t = target if isinstance(target, Tensor) else Tensor(target)

    lp = log_softmax(logits_t)
    return -(y_t * lp).sum()    


