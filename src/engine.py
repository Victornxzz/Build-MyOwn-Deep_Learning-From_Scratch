class Value :

    def __init__(self, data, _children=(), _op=""):

        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def _make(self, data, _children, _op): return type(self)(data, _children, _op)

    def __repr__(self):
        return f"Value(data={self.data}, op='{self._op}')"

    def __add__(self, other):
        other = other if isinstance(other, Value) else type(self)(other)
        return self._make(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        other = other if isinstance(other, Value) else type(self)(other)
        return self._make(self.data * other.data, (self, other), '*')

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only supporting int/float powers for now"
        return self._make(self.data ** exponent, (self,), f"**{exponent}")

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else type(self)(other)
        return other - self

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else type(self)(other)
        return other / self

def trace(root):

    nodes = set()
    edges = set()

    def build(v):

        if v not in nodes:

            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges