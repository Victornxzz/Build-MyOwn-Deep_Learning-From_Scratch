from typing import Any, Callable, List, Set, Tuple, Union
import numpy as np
from src.engine import Value

Operand = Union["Tensor", float, int, np.ndarray, List]

class Tensor:
    def __init__(
        self, 
        data: Operand, 
        _prev: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        
        if isinstance(data, Tensor):
            self.data: np.ndarray = np.ndarray(data.data, dtype=np.float64)
        else:
            self.data: np.ndarray = np.asarray(data, dtype=np.float64)       

        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._prev: Tuple[Tensor, ...] = _prev 
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def _make_tensor(self, *args, **kwargs)-> "Tensor":
        return type(self)(*args, **kwargs)

    def _coerce(self, other: Operand)-> "Tensor":
        return other if isinstance(other, Tensor) else self._make_tensor(other)

    @classmethod
    def from_value(cls, v: Any) -> "Tensor":
        data = v.data if hasattr(v, "data") else v
        return cls(np.array(data, dtype=np.float64))

    @property
    def shape(self)-> Tuple[int, ...]:
        return self.data.shape

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = shape[0]
        else:
            target_shape = shape

        out = self._make_tensor(
            self.data.reshape(target_shape),
            _prev=(self,),
            _op="reshape",
        )

        def _backward():
            Tensor._accumulate(self, out.grad.reshape(self.shape))
        
        out._backward = _backward
        return out

    def transpose(self)-> "Tensor":
        out = self._make_tensor(
            self.data.T,
            _prev=(self,),
            _op="transpose",
        )
        def _backward():
            Tensor._accumulate(self, out.grad.T)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @staticmethod
    def _accumulate(grad_into: "Tensor", incoming: np.ndarray) -> None:
        if grad_into.shape == ():
            grad_into.grad += np.sum(incoming)
        else:
            grad_into.grad += incoming

    def __add__(self, other: Operand) -> "Tensor":
        other_t = self._coerce(other)
        out = self._make_tensor(
            self.data + other_t.data,
            _prev=(self, other_t),
            _op ="+",
        )

        def _backward():
            Tensor._accumulate(self, out.grad)
            Tensor._accumulate(other_t, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Operand) -> "Tensor":
        other_t = self._coerce(other)
        out = self._make_tensor(
            self.data * other_t.data,
            _prev=(self, other_t),
            _op="*",
        )

        def _backward():
            Tensor._accumulate(self, out.grad * other_t.data)
            Tensor._accumulate(other_t, out.grad * self.data)

        out._backward = _backward
        return out

    def __pow__(self, c: Union[int, float])-> "Tensor":
        assert isinstance(c, (int, float)), f"Exponent must be numeric constant, got {type(c)}"
        out = self._make_tensor(
            self.data ** c,
            _prev=(self,),
            _op="**",
        )

        def _backward():
            local_grad = c * (self.data ** (c - 1.0))
            Tensor._accumulate(self, out.grad * local_grad)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = self._make_tensor(
            np.maximum(0.0, self.data),
            _prev=(self,),
            _op="relu",
        )

        def _backward():
            local_grad = (self.data > 0.0).astype(np.float64)
            Tensor._accumulate(self, out.grad * local_grad)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = self._make_tensor(
            t,
            _prev=(self,),
            _op="tanh",
        )

        def _backward():
            local_grad = 1.0 - t ** 2
            Tensor._accumulate(self, out.grad * local_grad)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = self._make_tensor(
            e,
            _prev=(self,),
            _op="exp",
        )

        def _backward():
            Tensor._accumulate(self, out.grad * e)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = self._make_tensor(
            np.log(self.data),
            _prev=(self,),
            _op="log",
        )

        def _backward():
            Tensor._accumulate(self, out.grad / self.data)

        out._backward = _backward
        return out

    def __matmul__(self, other: Operand) -> "Tensor":
        other_t = self._coerce(other)
        out_data = self.data @ other_t.data
        out = self._make_tensor(
            out_data,
            _prev=(self, other_t),
            _op="@",
        )

        def _backward():
            if self.data.ndim == 2 and other_t.data.ndim == 2:
                Tensor._accumulate(self, out.grad @ other_t.data.T)
                Tensor._accumulate(other_t, self.data.T @ out.grad)
                return

            a_mat = self.data.reshape(1, -1) if self.data.ndim == 1 else self.data
            b_mat = other_t.data.reshape(-1, 1) if other_t.data.ndim == 1 else other_t.data

            if self.data.ndim == 1 and other_t.data.ndim == 1:
                g_mat = np.asarray(out.grad).reshape(1, 1)
            elif self.data.ndim == 1:
                g_mat = out.grad.reshape(1 , -1)
            elif other_t.data.ndim == 1:
                g_mat = out.grad.reshape(-1, 1)
            else:
                g_mat = out.grad

            grad_a = g_mat @ b_mat.T
            grad_b = a_mat.T @ g_mat

            Tensor._accumulate(self, grad_a.reshape(self.shape))
            Tensor._accumulate(other_t, grad_b.reshape(other_t.shape))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: Operand) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Operand) -> "Tensor":
        return self._coerce(other) + (-self)

    def __truediv__(self, other: Operand) -> "Tensor":
        return self * (self._coerce(other) ** -1.0)

    def __rtruediv__(self, other: Operand) -> "Tensor":
        return self._coerce(other) * (self ** -1.0)

    def __radd__(self, other: Operand) -> "Tensor":
        return self + other

    def __rmul__(self, other: Operand) -> "Tensor":
        return self * other

    def backward(self) -> None:
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"