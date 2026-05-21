import numpy as np

from utils.metrics import evaluate


def test_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    r = evaluate(y, s)
    assert r.acc == 1.0
    assert abs(r.auc - 1.0) < 1e-6
    assert r.eer < 1e-6


def test_random_smoke():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(size=200)
    r = evaluate(y, s)
    assert 0.0 <= r.acc <= 1.0
    assert 0.0 <= r.auc <= 1.0
    assert 0.0 <= r.eer <= 1.0
