from typing import Sequence, Optional
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
        self.W: Tensor = Tensor(w_data)

        self.b: Tensor = Tensor(0.0)

    def __call__(self, x) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        z = x_tensor @ self.W + self.b

        if self.activation == "tanh":
            return z.tanh()
        elif self.activation == "relu":
            return z.relu()
        elif self.activation == "none":
            return z
        else : 
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> None: 
        return f"Neuron(n_in={self.n_in}, activation='{self.activation}')"


class Dense:

    def __init__(
            self,
            n_in: int,
            n_out: int,
            bias: bool = True,
            seed: Optional[int] = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = bias

        rng = np.random.default_rng(seed)

        w_data = rng.uniform(-1.0, 1.0, size=(n_in, n_out)).astype(np.float64)
        self.W: Tensor = Tensor(w_data)

        if self.use_bias:
            b_data = np.zeros((n_out), dtype=np.float64)
            self.b = Tensor(b_data)
        else:
            self.b = None

    def __call__(self, x) -> Tensor:

        if not isinstance(x, Tensor):
            raise TypeError(f"Expected input x to be a Tensor, got {type(x)}")

        is_1d = x.ndim == 1

        x_2d = x.reshape(1, self.n_in) if is_1d else x
        z = x_2d @ self.W

        if self.use_bias and self.b is not None:
            z = z + self.b

        return z.reshape(self.n_out) if is_1d else z

    def parameters(self) -> list[Tensor]:

        if self.use_bias and self.b is not None:
            return [self.W, self.b]
        return [self.W]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"Dense(n_in={self.n_in}, n_out={self.n_out}, bias={self.use_bias})"


class MLP:
    def __init__(
            self,
            sizes: Sequence[int],
            activation: str = "tanh",
            out_activation: str = "none",
            seed: Optional[int] = None,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("MLP demands in and out sizes (len >= 2)")

        valid_acts = {"tanh", "relu", "none"}
        if activation not in valid_acts or out_activation not in valid_acts:
            raise ValueError(f"Activations must be include one of {valid_acts}")

        self.sizes = sizes
        self.activation = activation
        self.out_activation = out_activation
        self.layers = []

        for i in range(len(sizes) - 1):
            s = seed + i if seed is not None else None
            self.layers.append(Dense(sizes[i], sizes[i + 1], bias=True, seed=s))

    @staticmethod
    def _apply_activation(z: Tensor, name: str) -> Tensor:
        if name == "tanh":
            return z.tanh()
        elif name == "relu":
            return z.relu()
        elif name == "none":
            return z
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            act_name = self.out_activation if i == len(self.layers) - 1 else self.activation
            out = self._apply_activation(out, act_name)
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"MLP(sizes={self.sizes}, activation='{self.activation}', out_activation='{self.out_activation}')"
