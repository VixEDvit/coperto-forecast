
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, HORIZON, make_features


def make_history(n_days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_days, freq="D"),
        "restaurant_id": 1,
        "guests": rng.poisson(150, size=n_days).astype(float),
        "is_closed": 0,
    })


def test_features_do_not_see_the_future():
    history = make_history()
    cutoff = history["date"].iloc[80]  # день T

    changed = history.copy()
    changed.loc[changed["date"] >= cutoff, "guests"] = 10_000

    before = make_features(history)
    after = make_features(changed)

    unaffected = before["date"] < cutoff + pd.Timedelta(days=HORIZON)
    pd.testing.assert_frame_equal(
        before.loc[unaffected, FEATURE_COLUMNS],
        after.loc[unaffected, FEATURE_COLUMNS],
    )


def test_lag_7_is_value_one_week_ago():
    features = make_features(make_history())
    expected = features["guests"].shift(7)
    pd.testing.assert_series_equal(features["lag_7"], expected, check_names=False)


def test_target_is_not_a_feature():
    assert "guests" not in FEATURE_COLUMNS


def test_closed_days_are_nan_in_lags():
    history = make_history()
    history.loc[10, ["guests", "is_closed"]] = [0, 1]
    features = make_features(history)
    assert np.isnan(features.loc[17, "lag_7"])


def test_missing_dates_raise_clear_error():
    history = make_history().drop(index=50)
    with pytest.raises(ValueError, match="пропуски"):
        make_features(history)