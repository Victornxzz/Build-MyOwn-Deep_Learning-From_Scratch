import random
import pytest
import numpy as np
import math
from src.engine import Value
from src.vec import Vec, Mat
from src.tensor import Tensor
from src.nn import Neuron, Dense, MLP
from src.loss import mse_loss, mae_loss, log_softmax, softmax, cross_entropy_loss
from src.optim import Optimizer, SGD, SGDMomentum, Adam, AdamW, RMSProp
from src.trainer import (
    accuracy, 
    train, 
    plot_history, 
    iterate_minibatches, 
    train_minibatch, 
    gradient_noise, 
    epochs_to_threshold,
    plot_batch_comparison
)
from src.init import (
    xavier_uniform,
    xavier_normal,
    he_normal,
    he_uniform,
    init_dense,
    forward_activation_stats,
)

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-4

# --- Small helpers -----------------------------------------------------------
def as_array(t):
    """Return the underlying numpy array of a Tensor (or pass arrays through)."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def central_diff(f, x, eps=EPS):
    """Numerical gradient of scalar-valued f at numpy point x (any shape)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def toy_data(n=40, seed=0, noise=0.1):
    """A small reproducible moons dataset (local fixture): X (n, 2), y (n,) in {-1, +1}."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0.0, np.pi, n_out)
    x_out = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    t_in = np.linspace(0.0, np.pi, n_in)
    x_in = np.stack([1.0 - np.cos(t_in), 0.5 - np.sin(t_in)], axis=1)
    X = np.concatenate([x_out, x_in], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(n_out), -np.ones(n_in)]).astype(np.float64)
    X += rng.normal(0.0, noise, size=X.shape)
    return Tensor(X), Tensor(y)


# ===========================================================================
# iterate_minibatches: exact partitioning
# ===========================================================================
def test_batches_cover_every_row_once_no_shuffle():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    batches = list(iterate_minibatches(X, y, batch_size=3, shuffle=False))
    # ceil(10 / 3) == 4 batches; sizes 3,3,3,1
    assert [b[0].shape[0] for b in batches] == [3, 3, 3, 1], "wrong batch sizes"
    Xcat = np.concatenate([b[0] for b in batches], axis=0)
    ycat = np.concatenate([b[1] for b in batches], axis=0)
    assert np.array_equal(Xcat, X), "no-shuffle batches must recover X in order"
    assert np.array_equal(ycat, y), "no-shuffle batches must recover y in order"


def test_drop_last_drops_partial_batch():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    batches = list(
        iterate_minibatches(X, y, batch_size=3, shuffle=False, drop_last=True)
    )
    assert len(batches) == 10 // 3, "drop_last must yield N // B batches"
    assert all(b[0].shape[0] == 3 for b in batches), "kept batches are full-size"


def test_shuffle_is_a_permutation_and_aligned():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = X[:, 0].copy()  # y[i] == X[i, 0] so we can check alignment after shuffle
    batches = list(iterate_minibatches(X, y, batch_size=5, shuffle=True, seed=1))
    Xcat = np.concatenate([b[0] for b in batches], axis=0)
    ycat = np.concatenate([b[1] for b in batches], axis=0)
    # All 12 rows present exactly once (as a set of first coords).
    assert sorted(Xcat[:, 0].tolist()) == sorted(X[:, 0].tolist()), (
        "shuffled batches must be a permutation covering every row once"
    )
    # X_b and y_b stayed paired through the shuffle.
    assert np.array_equal(ycat, Xcat[:, 0]), "shuffle must keep X_b and y_b aligned"


def test_shuffle_seed_is_reproducible_and_varies():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    a = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=7)]
    )
    b = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=7)]
    )
    c = np.concatenate(
        [b[1] for b in iterate_minibatches(X, y, 4, shuffle=True, seed=8)]
    )
    assert np.array_equal(a, b), "same seed must give the same order"
    assert not np.array_equal(a, c), "different seed should (almost surely) differ"


def test_bad_batch_size_raises():
    X = np.zeros((5, 2))
    y = np.zeros(5)
    for bad in (0, -1, 6):
        with pytest.raises(AssertionError):
            list(iterate_minibatches(X, y, batch_size=bad))


def test_mismatched_lengths_raise():
    with pytest.raises(AssertionError):
        list(iterate_minibatches(np.zeros((5, 2)), np.zeros(4), batch_size=2))


# ===========================================================================
# train_minibatch: step counts, history, and full-batch equivalence
# ===========================================================================
def test_step_count_matches_batches_per_epoch():
    X, y = toy_data(n=40, seed=0)
    model = MLP([2, 8, 1], activation="tanh", seed=0)
    epochs, B = 3, 16
    out = train_minibatch(
        model, X, y, lr=0.05, epochs=epochs, batch_size=B, shuffle=True, seed=0
    )
    n_batches = int(np.ceil(X.shape[0] / B))
    assert out["steps"] == epochs * n_batches, "steps == epochs * ceil(N/B)"
    assert len(out["batch_loss"]) == out["steps"], "one batch_loss per step"
    assert len(out["epoch_loss"]) == epochs, "one epoch_loss per epoch"


def test_epoch_loss_decreases():
    X, y = toy_data(n=60, seed=1)
    model = MLP([2, 16, 1], activation="tanh", seed=1)
    out = train_minibatch(
        model, X, y, lr=0.1, epochs=80, batch_size=16, shuffle=True, seed=1
    )
    hist = out["epoch_loss"]
    assert hist[-1] < hist[0], "epoch loss must fall over training"
    assert hist[-1] < 0.6 * hist[0], "loss should drop meaningfully, not crawl"


def test_full_batch_matches_manual_loop():
    """batch_size == N with shuffle=False must equal a hand-rolled full-batch
    loop on the SAME initial model -- i.e. mini-batch reduces to stage_15."""
    X, y = toy_data(n=24, seed=2)

    # Model A: trained via train_minibatch in full-batch mode.
    model_a = MLP([2, 8, 1], activation="tanh", seed=3)
    out = train_minibatch(
        model_a, X, y, lr=0.1, epochs=10, batch_size=X.shape[0], shuffle=False
    )

    # Model B: identical init, manual full-batch loop using the stage_15 stack.
    model_b = MLP([2, 8, 1], activation="tanh", seed=3)
    opt = SGD(model_b.parameters(), 0.1)
    manual = []
    yb = np.asarray(y.data, dtype=float).reshape(-1, 1)
    for _ in range(10):
        pred = model_b(X)
        loss = mse_loss(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        manual.append(float(loss.data))

    assert np.allclose(out["epoch_loss"], manual, atol=ATOL, rtol=RTOL), (
        "full-batch train_minibatch must match a manual full-batch loop:\n "
        f"train_minibatch={out['epoch_loss']}\n manual         ={manual}"
    )
    assert out["steps"] == 10, "full-batch => one step per epoch"


# ===========================================================================
# gradient_noise: Var[g_hat_B] ~ sigma**2 / B
# ===========================================================================
def test_gradient_noise_decreases_with_batch_size():
    X, y = toy_data(n=128, seed=4)
    model = MLP([2, 16, 1], activation="tanh", seed=5)
    noise_small = gradient_noise(model, X, y, batch_size=8, n_batches=40, seed=0)
    noise_large = gradient_noise(model, X, y, batch_size=64, n_batches=40, seed=0)
    assert noise_small > noise_large, (
        "gradient noise must shrink as batch size grows (sigma**2 / B):\n "
        f"B=8 -> {noise_small}, B=64 -> {noise_large}"
    )
    assert noise_large >= 0.0, "variance is non-negative"


def test_gradient_noise_does_not_move_model():
    X, y = toy_data(n=64, seed=6)
    model = MLP([2, 8, 1], activation="tanh", seed=7)
    before = [as_array(p).copy() for p in model.parameters()]
    _ = gradient_noise(model, X, y, batch_size=16, n_batches=10, seed=0)
    after = [as_array(p) for p in model.parameters()]
    for b, a in zip(before, after):
        assert np.allclose(b, a, atol=0.0), "gradient_noise must not update params"


# ===========================================================================
# epochs_to_threshold
# ===========================================================================
def test_epochs_to_threshold_first_crossing():
    hist = [1.0, 0.8, 0.5, 0.3, 0.25]
    assert epochs_to_threshold(hist, 0.55) == 3, "1-based index of first <= thr"
    assert epochs_to_threshold(hist, 1.0) == 1, "epoch 1 already meets threshold"
    assert epochs_to_threshold(hist, 0.1) == -1, "never reached => -1"


# ===========================================================================
# gradcheck: batched mse_loss backward vs central differences
# ===========================================================================
def test_batched_mse_backward_matches_central_diff():
    """One batched forward/backward through the MLP must produce a .grad equal to
    a central-difference gradient of the scalar batch loss w.r.t. a parameter."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2))            # a batch of B=6
    yb = rng.choice([-1.0, 1.0], size=(6, 1))  # targets, shaped to broadcast

    model = MLP([2, 5, 1], activation="tanh", seed=11)
    params = model.parameters()
    p = params[0]  # gradcheck the first parameter tensor (the first layer's W)
    p0 = as_array(p).copy()

    # Analytic grad from autodiff.
    model.zero_grad() if hasattr(model, "zero_grad") else None
    for q in params:
        q.grad = np.zeros_like(as_array(q))
    loss = mse_loss(model(Tensor(X)), yb)
    loss.backward()
    g_analytic = as_array(p.grad)

    # Numeric grad: perturb p.data entrywise, recompute the forward loss only.
    def loss_at(p_data):
        p.data = np.asarray(p_data, dtype=float).reshape(p0.shape)
        out = mse_loss(model(Tensor(X)), yb)
        return float(as_array(out))

    g_num = central_diff(loss_at, p0.copy())
    p.data = p0  # restore

    assert g_analytic.shape == g_num.shape, "grad shape must match param shape"
    assert np.allclose(g_analytic, g_num, atol=ATOL, rtol=RTOL), (
        "autodiff grad of batched mse_loss must match central differences:\n "
        f"analytic=\n{g_analytic}\n numeric=\n{g_num}"
    )