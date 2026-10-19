import numpy as np
from src.tensor import Tensor


class Neuron:

    def __init__(self, n_in: int, activation: str="tanh", seed: int | None = None):
        valid_activations = {"tanh", "relu", "none"}
        if activation not in valid_activations:
            raise ValueError(f"Activation '{activation}' not supported. Must be one of {valid_activations}")

        self.n_in = n_in
        self.activation = activation

        rng = np.random.default_rng(seed)
        w_data = rng.uniform(-1.0, 1.0, size=(n_in)).astype(np.float64)
        self.w: Tensor = Tensor(w_data)

        self.b: Tensor = Tensor(0.0)

    def __call__(self, x) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        z = x_tensor @ self.w + self.b

        if self.activation == "tanh":
            return z.tanh()
        elif self.activation == "relu":
            return z.relu()
        elif self.activation == "none":
            return z
        else : 
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> None: 
        return f"Neuron(n_in={self.n_in}, activation='{self.activation}')"

    