"""Tests for Stage 34: Capstone -- CIFAR-10.

Verifies the CIFAR-10 capstone built on top of the custom autograd and neural net engine.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

# Adiciona a raiz do projeto ao sys.path para garantir importações limpas de `src`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Importações diretas dos módulos do projeto
try:
    from src.engine import Value
    from src.tensor import Tensor, Vec
    from src.nn import (
        Adam,
        Augment,
        BatchNorm2d,
        ConvNet,
        Conv2D,
        accuracy,
        cosine_lr,
        cross_entropy_loss,
        make_cifar_like,
        normalize,
        random_crop,
        random_horizontal_flip,
        step_lr,
        train_cifar,
        DataLoader,
        Dataset,
    )
except (ImportError, ModuleNotFoundError) as exc:
    # Se os módulos ainda estiverem em desenvolvimento, faz skip limpo
    pytest.skip(
        f"Módulos do estágio 34 ou dependências prévias ainda não implementados: {exc}",
        allow_module_level=True,
    )

RNG = np.random.default_rng(34)
EPS = 1e-5
ATOL_BN = 1e-5
ATOL_E2E = 1e-4
RTOL = 1e-3


# --- Helpers de compatibilidade ---------------------------------------------
def as_array(t):
    """Retorna o ndarray subjacente de um Tensor ou repassa o próprio array."""
    return np.asarray(t.data if hasattr(t, "data") else t, dtype=float)


def make_tensor(arr):
    """Instancia um Tensor a partir de um array NumPy."""
    arr = np.asarray(arr, dtype=float)
    try:
        return Tensor(arr, requires_grad=True)
    except TypeError:
        return Tensor(arr)


def scalar(t):
    """Extrai um float escalar de um Tensor de tamanho 1 ou retorna a soma."""
    a = as_array(t)
    return float(a.reshape(-1)[0]) if a.size == 1 else float(a.sum())


def skip_if_unbuilt(fn, *args, **kwargs):
    """Executa a função e dá skip caso encontre um NotImplementedError de esqueleto."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"Esqueleto ainda não implementado: {exc}")


# ---------------------------------------------------------------------------
# BatchNorm2d forward: train vs eval
# ---------------------------------------------------------------------------
def test_batchnorm2d_train_standardizes_per_channel():
    """No modo de treino, cada canal de saída deve ter média ~0 e variância ~1."""
    x = RNG.standard_normal((8, 3, 5, 5)) * 2.0 + 1.0
    bn = skip_if_unbuilt(BatchNorm2d, 3)
    out = skip_if_unbuilt(bn, make_tensor(x))
    assert out.shape == (8, 3, 5, 5), f"Shape alterado: {out.shape}"
    o = as_array(out)
    
    mean_c = o.mean(axis=(0, 2, 3))
    var_c = o.var(axis=(0, 2, 3))
    np.testing.assert_allclose(
        mean_c, np.zeros(3), atol=1e-6,
        err_msg="BatchNorm2d train output deve ter média zero por canal.",
    )
    np.testing.assert_allclose(
        var_c, np.ones(3), atol=1e-3,
        err_msg="BatchNorm2d train output deve ter variância unitária por canal.",
    )


def test_batchnorm2d_train_updates_running_buffers():
    """O forward em modo train deve atualizar running_mean e running_var."""
    bn = skip_if_unbuilt(BatchNorm2d, 3, momentum=0.5)
    rm0 = np.array(bn.running_mean, dtype=float)
    rv0 = np.array(bn.running_var, dtype=float)
    x = RNG.standard_normal((6, 3, 4, 4)) + 3.0
    skip_if_unbuilt(bn, make_tensor(x))
    assert not np.allclose(bn.running_mean, rm0), "running_mean deve ser atualizado no modo train."
    assert not np.allclose(bn.running_var, rv0), "running_var deve ser atualizado no modo train."


def test_batchnorm2d_eval_uses_buffers_no_update():
    """O forward em modo eval deve usar os buffers fixos e não alterá-los."""
    bn = skip_if_unbuilt(BatchNorm2d, 2)
    skip_if_unbuilt(bn.eval)
    rm0 = np.array(bn.running_mean, dtype=float)
    rv0 = np.array(bn.running_var, dtype=float)
    x = RNG.standard_normal((5, 2, 3, 3))
    out = skip_if_unbuilt(bn, make_tensor(x))
    assert out.shape == (5, 2, 3, 3)
    np.testing.assert_allclose(
        bn.running_mean, rm0, atol=1e-12,
        err_msg="eval forward NÃO deve alterar running_mean.",
    )
    np.testing.assert_allclose(
        bn.running_var, rv0, atol=1e-12,
        err_msg="eval forward NÃO deve alterar running_var.",
    )


def test_batchnorm2d_parameters_excludes_buffers():
    """parameters() deve conter apenas [gamma, beta], excluindo buffers."""
    bn = skip_if_unbuilt(BatchNorm2d, 4)
    params = skip_if_unbuilt(bn.parameters)
    assert len(params) == 2, "BatchNorm2d.parameters() deve retornar [gamma, beta]."
    for p in params:
        assert as_array(p).shape == (4,), "gamma e beta devem ter shape (C,)."


# ---------------------------------------------------------------------------
# BatchNorm2d backward: gradcheck analítico vs numérico
# ---------------------------------------------------------------------------
def _bn_loss(bn, xarr, w):
    out = bn(make_tensor(xarr))
    return (out * make_tensor(w)).sum()


def test_batchnorm2d_gradcheck_input():
    shape = (4, 2, 3, 3)
    x0 = RNG.standard_normal(shape)
    w = RNG.standard_normal(shape)

    bn = skip_if_unbuilt(BatchNorm2d, 2)
    xt = make_tensor(x0)
    out = skip_if_unbuilt(bn, xt)
    loss = skip_if_unbuilt(lambda: (out * make_tensor(w)).sum())
    skip_if_unbuilt(loss.backward)
    g_analytic = np.array(xt.grad, dtype=float)

    g_num = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x0.copy(); xp[idx] += EPS
        xm = x0.copy(); xm[idx] -= EPS
        fp = scalar(_bn_loss(BatchNorm2d(2), xp, w))
        fm = scalar(_bn_loss(BatchNorm2d(2), xm, w))
        g_num[idx] = (fp - fm) / (2 * EPS)
        it.iternext()

    np.testing.assert_allclose(
        g_analytic, g_num, atol=ATOL_BN, rtol=RTOL,
        err_msg="Gradiente de entrada do BatchNorm2d não bate com as diferenças centrais.",
    )


def test_batchnorm2d_gradcheck_gamma_beta():
    shape = (5, 3, 2, 2)
    x0 = RNG.standard_normal(shape)
    w = RNG.standard_normal(shape)

    bn = skip_if_unbuilt(BatchNorm2d, 3)
    bn.gamma.data[:] = RNG.standard_normal(3) * 0.5 + 1.0
    bn.beta.data[:] = RNG.standard_normal(3) * 0.5
    gamma0 = np.array(bn.gamma.data, dtype=float)
    beta0 = np.array(bn.beta.data, dtype=float)

    out = skip_if_unbuilt(bn, make_tensor(x0))
    loss = skip_if_unbuilt(lambda: (out * make_tensor(w)).sum())
    skip_if_unbuilt(loss.backward)
    g_gamma = np.array(bn.gamma.grad, dtype=float)
    g_beta = np.array(bn.beta.grad, dtype=float)

    def loss_with_params(gamma_arr, beta_arr):
        b = BatchNorm2d(3)
        b.gamma.data[:] = gamma_arr
        b.beta.data[:] = beta_arr
        return scalar(_bn_loss_param(b, x0, w))

    num_gamma = np.zeros(3)
    num_beta = np.zeros(3)
    for c in range(3):
        gp = gamma0.copy(); gp[c] += EPS
        gm = gamma0.copy(); gm[c] -= EPS
        num_gamma[c] = (loss_with_params(gp, beta0) - loss_with_params(gm, beta0)) / (2 * EPS)
        bp = beta0.copy(); bp[c] += EPS
        bm = beta0.copy(); bm[c] -= EPS
        num_beta[c] = (loss_with_params(gamma0, bp) - loss_with_params(gamma0, bm)) / (2 * EPS)

    np.testing.assert_allclose(
        g_gamma, num_gamma, atol=ATOL_BN, rtol=RTOL,
        err_msg="Gradiente de gamma do BatchNorm2d não bate com as diferenças centrais.",
    )
    np.testing.assert_allclose(
        g_beta, num_beta, atol=ATOL_BN, rtol=RTOL,
        err_msg="Gradiente de beta do BatchNorm2d não bate com as diferenças centrais.",
    )


def _bn_loss_param(bn, xarr, w):
    out = bn(make_tensor(xarr))
    return (out * make_tensor(w)).sum()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def test_random_crop_preserves_shape():
    x = RNG.standard_normal((4, 3, 8, 8))
    out = skip_if_unbuilt(random_crop, x, pad=2, rng=np.random.default_rng(0))
    assert out.shape == x.shape, f"random_crop alterou a dimensão: {out.shape}"


def test_random_horizontal_flip_shape_and_flips_some():
    x = RNG.standard_normal((64, 3, 6, 6))
    out = skip_if_unbuilt(random_horizontal_flip, x, p=1.0, rng=np.random.default_rng(0))
    assert out.shape == x.shape
    np.testing.assert_allclose(
        out, x[:, :, :, ::-1], atol=1e-12,
        err_msg="p=1.0 deve inverter horizontalmente todas as imagens no eixo de largura.",
    )


def test_normalize_per_channel():
    x = RNG.standard_normal((10, 3, 5, 5))
    mean = [0.1, 0.2, 0.3]
    std = [2.0, 0.5, 1.5]
    out = skip_if_unbuilt(normalize, x, mean, std)
    ref = (x - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out, ref, atol=1e-12, err_msg="Incompatibilidade no normalize.")


def test_augment_eval_is_normalize_only_and_returns_array():
    x = RNG.standard_normal((6, 3, 8, 8))
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    aug = skip_if_unbuilt(Augment, pad=4, flip_p=0.5, mean=mean, std=std, seed=0)
    skip_if_unbuilt(aug.eval)
    out = skip_if_unbuilt(aug, x)
    assert isinstance(out, np.ndarray), "Augment deve retornar um ndarray do NumPy."
    np.testing.assert_allclose(
        out, x, atol=1e-12,
        err_msg="Augment em eval com média 0 e desvio 1 deve se comportar como a identidade.",
    )


# ---------------------------------------------------------------------------
# Learning Rate Schedules
# ---------------------------------------------------------------------------
def test_cosine_lr_endpoints_and_midpoint():
    base, mn, T = 0.1, 0.0, 100
    assert abs(skip_if_unbuilt(cosine_lr, 0, T, base_lr=base, min_lr=mn) - base) < 1e-12
    assert abs(skip_if_unbuilt(cosine_lr, T, T, base_lr=base, min_lr=mn) - mn) < 1e-9
    mid = skip_if_unbuilt(cosine_lr, T // 2, T, base_lr=base, min_lr=mn)
    assert abs(mid - 0.5 * base) < 1e-9, f"Ponto médio do cosine_lr deveria ser {0.5 * base}, obteve {mid}"


def test_cosine_lr_monotone_decreasing():
    T = 50
    vals = [skip_if_unbuilt(cosine_lr, t, T, base_lr=1.0) for t in range(T + 1)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(T)), "cosine_lr deve ser monotonamente decrescente."


def test_step_lr_drops():
    base, de, g = 1.0, 10, 0.1
    assert abs(skip_if_unbuilt(step_lr, 0, base_lr=base, drop_every=de, gamma=g) - 1.0) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 9, base_lr=base, drop_every=de, gamma=g) - 1.0) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 10, base_lr=base, drop_every=de, gamma=g) - 0.1) < 1e-12
    assert abs(skip_if_unbuilt(step_lr, 20, base_lr=base, drop_every=de, gamma=g) - 0.01) < 1e-12


# ---------------------------------------------------------------------------
# ConvNet forward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_shape,n_classes,channels", [
    ((3, 32, 32), 10, (4, 8)),
    ((3, 16, 16), 5, (4, 8)),
    ((1, 8, 8), 3, (2, 4)),
])
def test_convnet_forward_shape(in_shape, n_classes, channels):
    B = 3
    model = skip_if_unbuilt(
        ConvNet, in_shape, n_classes, channels=channels, hidden=8, seed=0
    )
    x = RNG.standard_normal((B, *in_shape))
    logits = skip_if_unbuilt(model, x)
    assert logits.shape == (B, n_classes), f"Esperava {(B, n_classes)}, obteve {logits.shape}"
    C, H, W = in_shape
    S = len(channels)
    expected_flat = channels[-1] * (H // (2 ** S)) * (W // (2 ** S))
    assert model.flat_dim == expected_flat, f"flat_dim deve ser deduzido ({expected_flat}); obteve {model.flat_dim}."


def test_convnet_parameters_nonempty_and_unique():
    model = skip_if_unbuilt(ConvNet, (3, 16, 16), 5, channels=(4, 8), seed=1)
    params = skip_if_unbuilt(model.parameters)
    assert len(params) >= 12, f"Esperava muitos parâmetros, obteve {len(params)}"
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameters() não deve repetir tensores."


def test_convnet_train_eval_propagates_to_batchnorm():
    model = skip_if_unbuilt(ConvNet, (3, 16, 16), 5, channels=(4, 8), seed=2)
    assert skip_if_unbuilt(model.eval) is model
    bns = [layer for layer in model.layers if isinstance(layer, BatchNorm2d)]
    assert bns, "ConvNet deve conter camadas BatchNorm2d."
    assert all(b.training is False for b in bns), "eval() deve propagar o estado para o BatchNorm."
    skip_if_unbuilt(model.train)
    assert all(b.training is True for b in bns), "train() deve propagar o estado para o BatchNorm."


# ---------------------------------------------------------------------------
# Gradcheck end-to-end na rede completa
# ---------------------------------------------------------------------------
def test_convnet_end_to_end_gradcheck():
    in_shape = (3, 8, 8)
    n_classes = 3
    B = 4
    model = skip_if_unbuilt(
        ConvNet, in_shape, n_classes, channels=(2, 3), hidden=4, seed=7
    )
    X = RNG.standard_normal((B, *in_shape))
    y = RNG.integers(0, n_classes, size=B)
    params = skip_if_unbuilt(model.parameters)

    def loss_value():
        model.zero_grad()
        logits = model(X)
        return cross_entropy_loss(logits, y)

    loss = skip_if_unbuilt(loss_value)
    skip_if_unbuilt(loss.backward)
    analytic = [np.array(p.grad, dtype=float) for p in params]

    rng = np.random.default_rng(123)
    for p, g_an in zip(params, analytic):
        flat = p.data.reshape(-1)
        n = flat.size
        probes = rng.choice(n, size=min(3, n), replace=False)
        for j in probes:
            orig = flat[j]
            flat[j] = orig + EPS
            fp = scalar(loss_value())
            flat[j] = orig - EPS
            fm = scalar(loss_value())
            flat[j] = orig
            num = (fp - fm) / (2 * EPS)
            an = g_an.reshape(-1)[j]
            assert abs(an - num) <= ATOL_E2E + RTOL * abs(num), (
                f"Falha no gradcheck end-to-end: analytic={an:.3e} "
                f"numeric={num:.3e} (param shape {p.data.shape}, idx {j})."
            )


# ---------------------------------------------------------------------------
# Métricas e convergência sintética
# ---------------------------------------------------------------------------
def test_accuracy_matches_numpy():
    logits = RNG.standard_normal((20, 5))
    y = RNG.integers(0, 5, size=20)
    ref = float((np.argmax(logits, axis=1) == y).mean())
    got = skip_if_unbuilt(accuracy, make_tensor(logits), y)
    assert abs(got - ref) < 1e-12, f"accuracy={got}, esperava {ref}"


def test_make_cifar_like_shapes():
    X, y = skip_if_unbuilt(
        make_cifar_like, n_per_class=6, img_size=16, n_classes=4, seed=0
    )
    assert X.shape == (24, 3, 16, 16), f"Formato inesperado em X: {X.shape}"
    assert y.shape == (24,), f"Formato inesperado em y: {y.shape}"
    assert set(np.unique(y).tolist()) == {0, 1, 2, 3}, "Os rótulos devem cobrir todas as classes."


@pytest.mark.slow
def test_train_cifar_converges_and_records_lr():
    X, y = skip_if_unbuilt(
        make_cifar_like, n_per_class=24, img_size=16, n_classes=3,
        noise=0.2, seed=3,
    )
    ds = skip_if_unbuilt(Dataset, X, y)
    loader = skip_if_unbuilt(DataLoader, ds, batch_size=12, shuffle=True, seed=3)
    model = skip_if_unbuilt(
        ConvNet, (3, 16, 16), 3, channels=(8, 16), hidden=32, seed=3
    )
    hist = skip_if_unbuilt(
        train_cifar, model, loader, epochs=8, base_lr=2e-3, schedule="cosine"
    )

    losses = hist["train_loss"]
    assert losses[-1] < losses[0], (
        f"A perda de treino deve diminuir: primeira={losses[0]:.3f} "
        f"última={losses[-1]:.3f}"
    )
    assert len(hist["lr"]) == hist["steps"], "O lr deve ser gravado a cada step."
    assert hist["lr"][0] >= hist["lr"][-1] - 1e-12, "O cosine_lr deve decair ao longo dos passos."

    model.eval()
    logits = skip_if_unbuilt(model, X)
    acc = skip_if_unbuilt(accuracy, logits, y)
    assert acc >= 0.85, f"Esperava acurácia de treino >= 0.85, obteve {acc:.3f}"