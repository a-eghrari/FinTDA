import numpy as np

from henon_map import HenonMap


def test_generate_time_series_shapes():
    model = HenonMap()
    x, y, a = model.generate_time_series(n_steps=5, use_cache=False)

    assert x.shape == (5,)
    assert y.shape == (5,)
    assert a.shape == (5,)


def test_generate_time_series_zero_steps():
    model = HenonMap()
    x, y, a = model.generate_time_series(n_steps=0)

    assert x.size == 0
    assert y.size == 0
    assert a.size == 0


def test_cache_matches_single_run():
    np.random.seed(123)
    cached_model = HenonMap()
    x10, y10, a10 = cached_model.generate_time_series(n_steps=10, use_cache=True)
    x20, y20, a20 = cached_model.generate_time_series(n_steps=20, use_cache=True)

    # First segment should be preserved.
    np.testing.assert_allclose(x10, x20[:10])
    np.testing.assert_allclose(y10, y20[:10])
    np.testing.assert_allclose(a10, a20[:10])

    np.random.seed(123)
    baseline_model = HenonMap()
    x_ref, y_ref, a_ref = baseline_model.generate_time_series(n_steps=20, use_cache=False)

    np.testing.assert_allclose(x20, x_ref)
    np.testing.assert_allclose(y20, y_ref)
    np.testing.assert_allclose(a20, a_ref)


def test_reset_cache_changes_series():
    np.random.seed(42)
    model = HenonMap()
    x1, y1, a1 = model.generate_time_series(n_steps=10, use_cache=True)
    model.reset_cache()
    x2, y2, a2 = model.generate_time_series(n_steps=10, use_cache=True)

    # With reset, cached data is cleared; sequences should differ most of the time.
    assert not np.allclose(x1, x2)
    assert not np.allclose(y1, y2)
    # a_values are deterministic given n_steps.
    assert np.allclose(a1, a2)
