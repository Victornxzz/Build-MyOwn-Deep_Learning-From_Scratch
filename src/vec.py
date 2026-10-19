import os
import sys
from typing import Iterable, List, Union,Tuple
from src.engine import Value

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
            return Vec([a + b for a, b in zip(self.data, other.data)])
        return Vec([a + other for a in self.data])            

    def __sub__(self, other: Union["Vec", Scalar]) -> "Vec":
        if isinstance(other, Vec):
            assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
            return Vec(a - b for a, b in zip(self.data, other.data))
        return Vec(a - other for a in self.data)

    def __mul__(self, other: Union["Vec", Scalar]) -> "Vec":
        if isinstance(other, Vec):
            assert len(self) == len(other), f"Lenght mismatch: {len(self)} vs {len(other)}"
            return Vec(a * b for a, b in zip(self.data, other.data))
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

class Mat:
    def __init__(self, data: Iterable[Iterable[Scalar]]):
        self.data: List[List[Value]] = []
        for row in data:
            new_row = [x if isinstance(x, Value) else Value(x) for x in row]
            self.data.append(new_row)

        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0

        for row in self.data:
            assert len(row) == self.cols

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, i: int) -> List[Value]:
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Mat(shape={self.shape})"

    def matmul(self, other: "Mat") -> "Mat":
        assert self.cols == other.rows, f"Inner-dim dismatch: {self.shape} @ {other.shape}"
        m, k, n = self.rows, self.cols, other.cols
        out_data = []
        for i in range(m):
            row_out = []
            for j in range(n):
                val = self.data[i][0] * other.data[0][j]
                for p in range(1, k):
                    val += (self.data[i][p] * other.data[p][j])
                row_out.append(val)
            out_data.append(row_out)
        return Mat(out_data)

    def __matmul__(self, other: "Mat") -> "Mat":
        return self.matmul(other)

    def transpose(self) -> "Mat":
        transposed_data = [
            [self.data[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]       
        return Mat(transposed_data) 

    @property
    def T(self) -> "Mat":
        return self.transpose()

    def reshape(self, rows: int, cols: int) -> "Mat":
        if rows * cols != self.rows * self.cols:
            raise ValueError(f"Cannot reshape matrix of size {self.rows * self.cols} into ({rows}, {cols})")
        flat = [val for row in self.data for val in row]

        reshaped_data = [
            flat[i * cols : (i + 1) * cols]
            for i in range(rows)
        ]
        return Mat(reshaped_data)

    def sum(self) -> Value:
        assert self.rows > 0 and self.cols > 0, "empty vector"
        flat = [val for row in self.data for val in row]
        result = flat[0]
        for val in flat[1:]:
            result += val
        return result

    def mean(self) -> Value:
        N = self.rows * self.cols
        return self.sum() * (1.0/N)