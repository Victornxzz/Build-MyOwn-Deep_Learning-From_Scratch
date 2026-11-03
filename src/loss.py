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

    m_val = np.max(logits_t.data, axis=-1, keepdims=True)
    m_tensor = Tensor(m_val)     
    shift = logits_t - m_tensor

    axis = 1 if len(logits_t.shape) == 2 else None
    lse = shift.exp().sum(axis=axis, keepdims=True).log()
    return shift - lse

def softmax(logits) -> Tensor:
    return log_softmax(logits).exp()

def cross_entropy_loss(logits, targets) -> Tensor:
    logits_t = logits if isinstance(logits, Tensor) else Tensor(logits)

    if isinstance(targets, Tensor):
        y_data = targets.data
    else:
        y_data = np.asarray(targets)

    if y_data.ndim == 1:
        if len(logits_t.shape) == 2:
            B, C = logits_t.shape
            one_hot = np.zeros((B, C), dtype=np.float64)
            one_hot[np.arange(B), y_data.astype(int)] = 1.0
            y_data = one_hot
        else:
            C = logits_t.shape[0]
            if y_data.size == 1:
                one_hot = np.zeros((C,), dtype=np.float64)
                one_hot[int(y_data)] = 1.0
                y_data = one_hot

    y_t = Tensor(y_data)
    lp = log_softmax(logits_t)

    axis = 1 if len(logits_t.shape) == 2 else None
    loss_per_row = -(y_t * lp).sum(axis=axis, keepdims=False)

    return loss_per_row.mean() 


