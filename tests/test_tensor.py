"""Tests for stage 2: Computational graph.

Validates that `Value` records its operation graph (`_prev`, `_op`) while
forward arithmetic stays identical to stage_01.

This stage does NOT implement a backward pass, so there are no analytical
`grad` fields to gradient-check yet. To still honor central-difference
gradient checking "where gradients exist", we treat the *forward* DAG as a
differentiable function of its leaf inputs and verify the local derivatives
the README derives ( d(a+b)/da = 1, d(a*b)/da = b ) numerically, by
re-running the forward graph with each leaf perturbed by +/-eps. When the
real backward pass arrives in a later stage, the SAME central-difference
recipe will be reused against analytical `.grad` values.
"""


import pytest
from src.engine import Value, trace, topo_sort

class _MOD:
    Value = Value
    trace = trace

TOL = 1e-5
EPS = 1e-6


def central_diff(f, x, eps=EPS):
    """Numerical derivative of scalar f at scalar x via central differences."""
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


def _f(x):
    return (x * x + 1.0) / (x - 4.0)
# ---------------------------------------------------------------------------
# Construction & repr
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------
# central-difference helper (reused verbatim by later stages vs analytical grad)
# --------------------------------------------------------------------------
def test_topo_sort_orders_parents_before_children():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    d = c + a
    order = topo_sort(d)
    assert set(order) == {a, b, c, d}, "topo must include every reachable node"
    assert len(order) == len(set(order)), "each node exactly once"
    pos = {v: i for i, v in enumerate(order)}
    assert pos[a] < pos[c] and pos[b] < pos[c], "parents of c before c"
    assert pos[c] < pos[d] and pos[a] < pos[d], "parents of d before d"


def test_topo_sort_terminates_on_reuse():
    a = Value(3.0)
    out = a * a
    order = topo_sort(out)
    assert set(order) == {a, out}
    assert len(order) == 2


# ---------------------------------------------------------------------------
# backward: seeds output grad = 1
# ---------------------------------------------------------------------------
def test_backward_seeds_output_with_one():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert out.grad == pytest.approx(1.0)


def test_backward_simple_mul():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert a.grad == pytest.approx(3.0)   # d(ab)/da = b
    assert b.grad == pytest.approx(2.0)   # d(ab)/db = a


# ---------------------------------------------------------------------------
# the canonical micrograd accumulation case
#   a=2, b=3, c=a*b, d=c+a, d.backward()  ->  a.grad==4, b.grad==2
# ---------------------------------------------------------------------------
def test_backward_accumulates_over_two_paths():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    d = c + a
    d.backward()
    assert a.grad == pytest.approx(4.0), "a reaches d via c (b=3) and directly (1) -> 4"
    assert b.grad == pytest.approx(2.0)


def test_backward_self_mul():
    a = Value(3.0)
    out = a * a            # d/da (a^2) = 2a = 6
    out.backward()
    assert a.grad == pytest.approx(6.0)


def test_backward_shared_subexpression():
    a = Value(3.0)
    b = a + 1.0            # 4
    c = a * 2.0            # 6
    out = b * c            # 24 ; d/da = (a+1)*2 + 2a = 4a+2 = 14
    out.backward()
    assert a.grad == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# gradcheck a composite expression against central differences
#   f(x) = (x*x + 1) / (x - 4)     (uses + * ** / and a constant)
# ---------------------------------------------------------------------------
def _f(x):
    return (x * x + 1.0) / (x - 4.0)


@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.0, 1.3, 2.7])
def test_backward_gradcheck_composite(x0):
    x = Value(x0)
    out = (x * x + 1.0) / (x - 4.0)
    out.backward()
    num = central_diff(_f, x0)
    assert x.grad == pytest.approx(num, abs=TOL, rel=1e-4), (
        f"backward dy/dx at x={x0}: analytic={x.grad}, numeric={num}"
    )


@pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 3.5])
def test_backward_gradcheck_with_pow(x0):
    # f(x) = x**3 - 2*x   ->  f'(x) = 3x^2 - 2
    x = Value(x0)
    out = x ** 3 - 2.0 * x
    out.backward()
    assert x.grad == pytest.approx(3.0 * x0 ** 2 - 2.0, abs=TOL, rel=1e-4)


# ---------------------------------------------------------------------------
# does not zero grads first (accumulates across calls if not reset)
# ---------------------------------------------------------------------------
def test_backward_does_not_zero_existing_grad():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    first = a.grad
    out.backward()  # second pass, no zeroing
    assert a.grad == pytest.approx(2.0 * first), "grad accumulates across backward calls"


# ---------------------------------------------------------------------------
# leaf closure is a no-op (inherited from stage_02)
# ---------------------------------------------------------------------------
def test_leaf_backward_noop():
    a = Value(4.0)
    a._backward()
    assert a.grad == 0.0


def test_repr_has_data_and_grad():
    a = Value(2.0)
    r = repr(a)
    assert "data=" in r and "grad=" in r