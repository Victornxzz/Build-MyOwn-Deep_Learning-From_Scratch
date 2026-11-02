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
            seed: int | None = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = bias

        rng = np.random.default_rng(seed)

        w_data = rng.uniform(-1.0, 1.0, size=(n_in, n_out)).astype(np.float64)
        self.W: Tensor = Tensor(w_data)

        if self.use_bias:
            b_data = np.zeros((n_out), dtype=np.float64)
            self.b: Tensor | None = Tensor(b_data)
        else:
            self.b = None

    def __call__(self, x) -> Tensor:

        if not isinstance(x, Tensor):
            raise TypeError(f"Expected input x to be a Tensor, got {type(x)}")
        
        is_1d = x.ndim == 1

        if is_1d:
            x_2d = x.reshape(1, self.n_in)
        else:
            x_2d = x

        z = x_2d @ self.W

        if self.use_bias and self.b is not None:
            batch_size = z.shape[0]
            ones = Tensor(np.ones((batch_size, 1), dtype=np.float64))
            b_row = self.b.reshape(1, self.n_out)
            bias_expanded = ones @ b_row
            z = z + bias_expanded

        if is_1d:
            return z.reshape(self.n_out)

        return z

    def parameters(self) -> list[Tensor]:

        if self.use_bias and self.b is not None:
            return [self.W, self.b]
        return [self.W]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"Dense(n_in={self.n_in}, n_out={self.n_out}, bias={self.use_bias})"
