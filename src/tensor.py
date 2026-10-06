import os
import sys
from typing import Iterable, List, Union
from engine import Value

Scalar = Union[int, float, Value]

class Vec:
    def __init__(self, data: Iterable[Scalar]) -> None:
        self.data: List[Value] = [x if isinstance(x, Value) else Value(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Value:
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Vec({[v.data for v in self.data]})"

    def __add__(self, other: Union["Vec", Scalar]) -> "Vec":
        if isinstance(other, Vec):
            assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
            return Vec(a + b for a, b in zip(self.data, other.data))
        return Vec(a + other for a in self.data)            

    def __sub__(self, other: Union["Vec", Scalar]) -> "Vec":
        if isinstance(other, Vec):
            assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
            return Vec(a - b for a, b in zip(self.data, other.data))
        return Vec(a - other for a in self.data)

    def __mul__(self, other: Union["Vec", Scalar]) -> "Vec":
        if isinstance(other, Vec):
            assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
            return Vec(a * b for a, b in self.data)
        return Vec(a * other for a in self.data)

    def __radd__(self, other: Scalar) -> "Vec":
        return self + other

    def __rsub__(self, other: Scalar) -> "Vec":
        return Vec(other - a for a in self.data)

    def __rmul__(self, other: Scalar) -> "Vec":
        return self * other

    def dot(self, other: "Vec") -> Value:
        assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
        result = self.data[0] * other.data[0]
        for a, b in zip(self.data[1:], other.data[1:]):
            result += (a * b)
        return result 

    def sum(self) -> Value:
        result = self.data[0]
        for v in self.data[1:]:
            result = result + v
        return result

    def relu(self) -> "Vec":
        return Vec(v.relu() for v in self.data)