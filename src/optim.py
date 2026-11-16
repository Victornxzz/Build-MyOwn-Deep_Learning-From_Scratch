from typing import Iterable, List, Tuple
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

class RMSProp(Optimizer):
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float = 1e-2,
            beta: float = 0.99,
            eps: float = 1e-8,
            weight_decay: float = 0.0
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.v = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            # Atualiza EMA do gradiente ao quadrado
            v = self.beta * self.v[i] + (1.0 - self.beta) * (g ** 2)
            self.v[i] = v

            # Atualização adaptativa in-place
            p.data -= self.lr * g / (np.sqrt(v) + self.eps)

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, beta={self.beta}, eps={self.eps}, weight_decay={self.weight_decay})"

class Adam(Optimizer):
    """Adam: EMA de 1º momento (m) + EMA de 2º momento (v) + correção de viés."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]

    def _effective_grad(self, p: Tensor) -> np.ndarray:
        if self.weight_decay != 0.0:
            return p.grad + self.weight_decay * p.data
        return p.grad

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = self._effective_grad(p)

            m = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            v = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)
            self.m[i] = m
            self.v[i] = v

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )


class AdamW(Adam):

    def _effective_grad(self, p: Tensor) -> np.ndarray:
        return p.grad

    def step(self) -> None:
        super().step()
        if self.weight_decay != 0.0:
            for p in self.params:
                if p.grad is None:
                    continue
                p.data -= self.lr * self.weight_decay * p.data

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )