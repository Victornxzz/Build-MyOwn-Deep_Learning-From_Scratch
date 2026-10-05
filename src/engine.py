from typing import Union, List, Set, Tuple
import math

Number = Union[int, float]

class Value :

    def __init__(self, data, _children=(), _op=""):

        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def _make(self, data : Number, _children : tuple = (), _op: str = "") -> "Value": 
        return type(self)(data, _children, _op)

    def __repr__(self):
        return f"Value(data={self.data}, grad='{self.grad}')"

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else self._make(other)
        out = self._make(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else type(self)(other)
        out = self._make(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent) -> "Value":
        assert isinstance(exponent, (int, float)), "only supporting int/float powers for now"
        out = self._make(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __truediv__(self, other) -> "Value":
        return self * (other ** -1)

    def __radd__(self, other) -> "Value":
        return self + other

    def __rmul__(self, other) -> "Value":
        return self * other

    def __rsub__(self, other) -> "Value":
        other = other if isinstance(other, Value) else type(self)(other)
        return other - self

    def __rtruediv__(self, other) -> "Value":
        other = other if isinstance(other, Value) else type(self)(other)
        return other / self

    def tanh(self) -> "Value":
        x = self.data
        t = math.tanh(x)
        out = self._make(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        x = self.data
        out = self._make(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = self._make(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = topo_sort(self) 
        self.grad = 1.0
        for node in reversed(topo): 
            node._backward()


def trace(root):

    nodes: set[Value] = set()
    edges: set[Value] = set()

    def build(v):

        if v not in nodes:

            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges

def topo_sort(root: Value) -> List[Value]:
    topo: List[Value] = []
    visited: Set[Value] = set()
    
    def build_topo(v: Value) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(root)
    return topo