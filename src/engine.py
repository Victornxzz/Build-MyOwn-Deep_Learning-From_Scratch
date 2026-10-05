from typing import Union

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