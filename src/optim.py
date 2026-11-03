from typing import Iterable, List
import numpy as np
from src.tensor import Tensor

class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params: List[Tensor] = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data, dtype=np.float64)

    def step(self) -> None:
        raise NotImplementedError("Optimizer.step must be implemented by a subclass")

class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning data (lr) must be > 0. lr = {lr}")

        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, n_params={len(self.params)})"

    