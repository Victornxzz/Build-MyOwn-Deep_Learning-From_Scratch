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


class SGDMomentum(SGD):
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float,
            beta: float = 0.0,
            nesterov: bool = False
    ) -> None:

        super().__init__(params, lr)

        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be between [0,1), received: {beta}")

        self.beta = beta
        self.nesterov = nesterov

        self.velocities = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            v = self.velocities[i]

            v = self.beta * v + g
            self.velocities[i] = v

            if self.nesterov:
                p.data -= self.lr * (g+ self.beta * v)
            else:
                p.data -= self.lr * v

    def reset(self) -> None:
        for i in range(len(self.velocities)):
            self.velocities[i].fill(0.0)

    def __repr__(self) -> str:
        return f"SGDMomentum(lr={self.lr}, beta={self.beta}, nesterov={self.nesterov})"

Momentum = SGDMomentum